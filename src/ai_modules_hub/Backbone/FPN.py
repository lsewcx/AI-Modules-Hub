import torchvision
import torch.nn as nn


class FPN(nn.Module):
    """
    论文地址:
        https://arxiv.org/abs/1612.03144
    """
    def torchvision_FPN(self, in_channels_list, out_channels, extra_blocks=None, norm_layer=None):
        return torchvision.ops.FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer)
