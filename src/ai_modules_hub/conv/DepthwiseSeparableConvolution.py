import torch.nn as nn

class DepthwiseSeparableConvolution(nn.Module):
    """
    Depthwise Separable Convolution 是一种高效的卷积操作，旨在减少计算量和参数数量。

    论文地址:
        https://arxiv.org/abs/1610.02357

    GitHub 地址:
        https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch/blob/master/DepthwiseSeparableConvolution/DepthwiseSeparableConvolution.py

    Depthwise Separable Convolution 由两个步骤组成:
        1. Depthwise Convolution: 对每个输入通道分别进行卷积。
        2. Pointwise Convolution: 使用 1x1 卷积将深度卷积的输出进行线性组合。

    优点:
        - Depthwise Separable Convolution 可以有效地减少计算量和参数数量，从而提高模型的计算效率。
        - 这种卷积操作可以有效地提高模型的泛化能力，减少过拟合的风险。
        - Depthwise Separable Convolution 可以很容易地集成到现有的卷积神经网络架构中，无需对原有网络进行大幅度修改。

    示例:
        >>> import torch
        >>> from ai_modules_hub.conv import DepthwiseSeparableConvolution
        >>> model = DepthwiseSeparableConvolution(in_channels=32, out_channels=64)
        >>> input_tensor = torch.randn(1, 32, 128, 128)
        >>> output_tensor = model(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 64, 128, 128])
    """

    def __init__(self, in_channels, out_channels):
        """初始化 DepthwiseSeparableConvolution 模块。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
        """
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == "__main__":
    import torch
    model = DepthwiseSeparableConvolution(in_channels=32, out_channels=64)
    input_tensor = torch.randn(1, 32, 128, 128)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)