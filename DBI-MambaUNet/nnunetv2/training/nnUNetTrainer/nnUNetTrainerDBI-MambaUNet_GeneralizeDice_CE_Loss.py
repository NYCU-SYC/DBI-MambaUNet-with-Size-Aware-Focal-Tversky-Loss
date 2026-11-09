from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from importlib import import_module
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import numpy as np
from nnunetv2.training.loss.dice import GeneralizedDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss


class nnUNetTrainerDBIMambaUNet_GeneralizedDice_CE_Loss(nnUNetTrainer):
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
            model = import_module("nnunetv2.nets.DBI-MambaUNet_3d").get_dbimambaunet_3d_from_plans(plans_manager, dataset_json, configuration_manager,
                                                                   num_input_channels,
                                                                   deep_supervision=enable_deep_supervision)
        else:
            raise NotImplementedError("Only 2D and 3D models are supported")

        print("DBI-MambaUNet: {}".format(model))

        return model

    def _build_loss(self):
        loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=self.label_manager.ignore_label, dice_class=GeneralizedDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

globals()["nnUNetTrainerDBI-MambaUNet_GeneralizedDice_CE_Loss"] = nnUNetTrainerDBIMambaUNet_GeneralizedDice_CE_Loss
globals()["nnUNetTrainerDBI-MambaUNet_GeneralizeDice_CE_Loss"] = nnUNetTrainerDBIMambaUNet_GeneralizedDice_CE_Loss
