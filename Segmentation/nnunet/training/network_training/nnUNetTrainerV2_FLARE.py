import torch
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import time

        
class nnUNetTrainerV2_FLARE_Big(nnUNetTrainerV2):
    def initialize_network(self):
        self.conv_per_stage = 3
        self.base_num_features = 32
        self.max_num_features = 512
        self.max_num_epochs = 50
        ##########
        #CO2eq:  12.523613659090 g\\\\\\\\\1\\\\
        #CO2eq:  524.076477098376 g\\\\50\\\\
        
        # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False
        
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, self.max_num_features)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_elastic"] = True

# ####计算参数量################################################################################
# import argparse
# from torchsummary import summary
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--name', type=str, default='version11')
# # parser.add_argument('--seed', type=int, default=10)
# # opts = parser.parse_args()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = nnUNetTrainerV2_FLARE_Big(22,all).network
# summary(model, (48, 224, 224))
# # summary(model, [(1, 256, 256),(1, 256, 256)])



class nnUNetTrainerV2_FLARE_Small(nnUNetTrainerV2):
    def initialize_network(self):
        self.conv_per_stage = 2
        self.stage_num = 5
        self.base_num_features = 16
        self.max_num_features = 256
        self.max_num_epochs = 1500
        #CO2eq:  3.341207060911 g/////////2
        # CO2eq:  13.118160474210 g///////10
        # CO2eq:  62.667218110995 g ////////50
        # CO2eq:  1868.793733338671 g//////1500

    #     Actual
    #     consumption:
    #     Time: 7:00: 30
    #     Energy: 3.934302596502
    #     kWh
    #     CO2eq: 1868.793733338671
    #     g
    #     This is equivalent
    #     to:
    #     17.384127751988
    #     km
    #     travelled
    #     by
    #     car
    #
    # CarbonTracker: Finished
    # monitoring.

    # 取消验证，加速训练
        self.num_val_batches_per_epoch = 1
        self.save_best_checkpoint = False
        

        if len(self.net_conv_kernel_sizes) > self.stage_num:
            self.net_conv_kernel_sizes = self.net_conv_kernel_sizes[:self.stage_num]
            self.net_num_pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes[:self.stage_num-1]

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, None,
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True, self.max_num_features)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
    def setup_DA_params(self):
        super().setup_DA_params()
        self.data_aug_params["do_mirror"] = False
        self.data_aug_params["do_elastic"] = True

# ####计算参数量################################################################################
# import argparse
# from torchsummary import summary
# # parser = argparse.ArgumentParser()
# # parser.add_argument('--name', type=str, default='version11')
# # parser.add_argument('--seed', type=int, default=10)
# # opts = parser.parse_args()
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = nnUNetTrainerV2_FLARE_Small(22,all)
# summary(model, (48, 224, 224))
# # summary(model, [(1, 256, 256),(1, 256, 256)])

