from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from torch import nn
from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
from nnunetv2.nets.UMambaBot_2d import get_umamba_bot_2d_from_plans
from importlib import import_module
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import torch
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import numpy as np


class nnUNetTrainerDBIMambaUNet_Dice_Loss(nnUNetTrainer):
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


globals()["nnUNetTrainerDBI-MambaUNet_Dice_Loss"] = nnUNetTrainerDBIMambaUNet_Dice_Loss
