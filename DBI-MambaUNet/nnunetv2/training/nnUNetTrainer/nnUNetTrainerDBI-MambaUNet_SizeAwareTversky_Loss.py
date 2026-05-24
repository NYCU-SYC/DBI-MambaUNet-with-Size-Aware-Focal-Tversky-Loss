from importlib import import_module

import numpy as np
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from nnunetv2.training.loss.compound_losses import SizeAwareTversky_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn


class nnUNetTrainerDBIMambaUNet_SizeAwareTversky_Loss(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   dataset_json,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        if len(configuration_manager.patch_size) == 2:
            model = get_umamba_bot_2d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                 num_input_channels, deep_supervision=enable_deep_supervision)
        elif len(configuration_manager.patch_size) == 3:
            model = import_module("nnunetv2.nets.DBI-MambaUNet_3d").get_dbimambaunet_3d_from_plans(
                plans_manager, dataset_json, configuration_manager, num_input_channels,
                deep_supervision=enable_deep_supervision
            )
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("DBI-MambaUNet: {}".format(model))
        return model

    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        focal_tversky_kwargs = {
            'alpha': 0.3,
            'beta': 0.7,
            'gamma': 1.0,
            'size_gamma': 0.8,
            'bg_weight': 0.8,
            'conn': 3,
            'normalize_pos': True,
            'smooth': 1e-6
        }
        ce_kwargs = {}

        loss = SizeAwareTversky_and_CE_loss(
            focal_tversky_kwargs=focal_tversky_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=0.0,
            weight_ft=1.0,
            ignore_label=self.label_manager.ignore_label
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


globals()["nnUNetTrainerDBI-MambaUNet_SizeAwareTversky_Loss"] = nnUNetTrainerDBIMambaUNet_SizeAwareTversky_Loss
