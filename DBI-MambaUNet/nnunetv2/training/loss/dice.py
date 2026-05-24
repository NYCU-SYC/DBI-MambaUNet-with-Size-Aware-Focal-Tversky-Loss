from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn

from typing import Callable

import torch
from torch import nn
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class GeneralizedDiceLoss(nn.Module):
    """
    Generalised Dice Loss (Sudre et al., 2017)

    主要公式：
        GDS = 2 * sum_l w_l * sum_n r_{ln} p_{ln} / sum_l w_l * sum_n (r_{ln} + p_{ln})
        w_l = 1 / ( sum_n r_{ln} )^vol_power   （預設 vol_power=2.0 → GDLv）

    這裡模組回傳的是 -GDS（與 SoftDiceLoss / FocalDiceLoss 風格一致），
    與論文定義的 GDL = 1 - GDS 僅差一個常數 1，不影響梯度與訓練。

    參數：
        apply_nonlin: 例如 softmax_helper_dim1
        batch_dice:   True 時在 batch 維度上一起計算一個 GDS
        do_bg:        是否將背景頻道納入（False 時忽略 channel 0）
        smooth:       穩定項，加在分母（實際用法：denom + smooth + eps）
        ddp:          若為 True 且 batch_dice=True，使用 AllGatherGrad 聚合多 GPU 統計量
        vol_power:    class weight 的體積指數，w_l ∝ 1 / vol_l^vol_power
        eps:          避免除以 0 的安全小值
    """

    def __init__(
        self,
        apply_nonlin: Callable = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 0.,
        ddp: bool = True,
        vol_power: float = 2.0,
        eps: float = 1e-8,
    ):
        super(GeneralizedDiceLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.vol_power = vol_power
        self.eps = eps

    def forward(self, x, y, loss_mask=None):
        """
        x: net_output, shape (B, C, ...)
        y: label map (B, ...) 或 one-hot (B, C, ...)
        loss_mask: (B, 1, ...) 有效區域 = 1，無效 = 0
        """
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # 空間維度
        axes = list(range(2, len(x.shape)))  # e.g. [2,3,4] for 3D

        # --- 建立 one-hot GT ---
        with torch.no_grad():
            if len(x.shape) != len(y.shape):
                # y: (B, ...) -> (B, 1, ...)
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # 已是 one-hot
                y_onehot = y
            else:
                gt = y.long()
                # 用 bool 或 uint8 都可以，與 MemoryEfficientSoftDiceLoss 相容
                y_onehot = torch.zeros_like(x, dtype=torch.bool, device=x.device)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # 預測也要去掉背景，注意這個動作要在 no_grad 外面才有 gradient
        if not self.do_bg:
            x = x[:, 1:]

        # --- 套用 loss_mask（若有） ---
        if loss_mask is not None:
            # loss_mask: (B, 1, ...) 會自動 broadcast 到 channel 維度
            x_masked = x * loss_mask
            y_masked = y_onehot * loss_mask
        else:
            x_masked = x
            y_masked = y_onehot

        # ========= batch_dice: 把 batch 也當成大 volume =========
        if self.batch_dice:
            # sum over batch + 空間維度 → (C,)
            reduce_axes = [0] + axes

            intersect = (x_masked * y_masked).sum(dim=reduce_axes)   # (C,)
            sum_pred = x_masked.sum(dim=reduce_axes)                 # (C,)
            sum_gt = y_masked.sum(dim=reduce_axes)                   # (C,)

            # 多 GPU DDP 時聚合
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            # class-wise 體積
            sum_gt_float = sum_gt.float()

            # w_l = 1 / (vol_l^vol_power)，空類別（vol=0）權重設為 0
            w = 1.0 / torch.clamp(sum_gt_float ** self.vol_power, min=self.eps)
            w = torch.where(sum_gt_float > 0, w, torch.zeros_like(w))

            # generalized dice score
            num = (w * intersect).sum()                        # scalar
            denom = (w * (sum_pred + sum_gt)).sum()           # scalar

            gds = (2.0 * num) / (denom + self.smooth + self.eps)
            loss = -gds  # 與其他 DiceLoss 風格一致

        # ========= 非 batch_dice: 每個 sample 各自一個 GDS =========
        else:
            # sum over 空間維度 → (B, C)
            intersect = (x_masked * y_masked).sum(dim=axes)   # (B, C)
            sum_pred = x_masked.sum(dim=axes)                 # (B, C)
            sum_gt = y_masked.sum(dim=axes)                   # (B, C)

            sum_gt_float = sum_gt.float()

            # w_{b,c} = 1 / vol_{b,c}^vol_power，空類別權重 0
            w = 1.0 / torch.clamp(sum_gt_float ** self.vol_power, min=self.eps)
            w = torch.where(sum_gt_float > 0, w, torch.zeros_like(w))  # (B, C)

            num = (w * intersect).sum(dim=1)                  # (B,)
            denom = (w * (sum_pred + sum_gt)).sum(dim=1)      # (B,)

            gds = (2.0 * num) / (denom + self.smooth + self.eps)  # (B,)
            loss = -gds.mean()

        return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = list(range(2, len(x.shape)))
        with torch.no_grad():
            if len(x.shape) != len(y.shape):
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
        sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

        if self.ddp and self.batch_dice:
            intersect = AllGatherGrad.apply(intersect).sum(0)
            sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

        if self.batch_dice:
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc




class FocalDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = True, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, version: int = 2, sample_ratio: float = 0.2, num_bins: int = 4, v1_weight=1.0,
                 v2_weight=1.0):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(FocalDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.sample_ratio = sample_ratio
        self.version = version
        self.num_bins = num_bins
        self.v1_weight = v1_weight
        self.v2_weight = v2_weight

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = list(range(2, len(x.shape)))
        with torch.no_grad():
            if len(x.shape) != len(y.shape):
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.int)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)

        tp = (x * y_onehot).flatten(1).sum(1)
        fp = (x * (1 - y_onehot)).flatten(1).sum(1)
        fn = ((1 - x) * y_onehot).flatten(1).sum(1)
        tn = ((1 - x) * (1 - y_onehot)).flatten(1)

        b_size = tn.shape[0]
        tn, _ = torch.sort(tn, descending=True, stable=True, dim=-1)
        tn_bins = torch.chunk(tn, chunks=self.num_bins, dim=-1)
        tn_bins = torch.cat(tn_bins, dim=1).reshape([tn.shape[0], self.num_bins, -1])
        with torch.no_grad():
            sample_num = int(tn.shape[-1] * self.sample_ratio / self.num_bins)
        sampled_tn = tn_bins[:, :, :sample_num]
        sampled_tn = sampled_tn.flatten(1).sum(1)

        # 初始化sum_pred，这样无论条件如何，它都有一个基本值
        sum_pred = torch.tensor(0.0, device=x.device)
        if self.batch_dice:
            # 这里的代码逻辑和之前一样
            intersect = intersect.sum(0)
            sum_gt = sum_gt.sum(0)
            if self.version == 1:
                # 确保 tp, fp, fn 和 sampled_tn 已经根据batch_dice计算
                sum_pred = (tp + fp + fn + sampled_tn).sum(0)
            elif self.version == 2:
                # 这里也是，确保 fp 和 sampled_tn 根据batch_dice计算
                sum_pred = (fp + sampled_tn).sum(0)
        else:
            if self.version == 1:
                sum_pred = tp + fp + fn + sampled_tn
            elif self.version == 2:
                sum_pred = fp + sampled_tn

            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
        dc = dc.mean()

        return -dc


class LesionSensitiveDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(LesionSensitiveDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + 2 * fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for i in range(2, len(tp.shape))]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)

    dl_focal = FocalDiceLoss(apply_nonlin=softmax_helper_dim1,batch_dice=True,do_bg=False,smooth=0,ddp=False,
                             sample_ratio=0.1,num_bins=1,version=2)

    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    res_focal = dl_focal(pred, ref)
    print(res_old, res_new, res_focal)
