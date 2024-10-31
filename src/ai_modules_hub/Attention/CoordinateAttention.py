import torch.nn as nn
import torch


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordinateAttention(nn.Module):
    """
    CoordinateAttention 是一种注意力机制，旨在增强卷积神经网络（CNN）的表示能力。

    论文地址:
        https://arxiv.org/pdf/2103.02907

    GitHub 源码地址:
        https://github.com/houqb/CoordAttention/blob/main/coordatt.py

    优点:
        - CoordinateAttention 可以有效地捕捉图像中的长距离依赖关系，这对于处理具有复杂结构的图像非常有用。
        - 通过分离通道注意力和空间注意力，CoordinateAttention 能够更好地保留和利用空间信息，从而提高模型的表示能力。
        - 相较于其他注意力机制（如自注意力机制），CoordinateAttention 更加轻量级，计算成本较低，适合在资源受限的环境中使用。
        - 这种注意力机制可以很容易地集成到现有的卷积神经网络架构中，无需对原有网络进行大幅度修改。
    """

    def __init__(self, inp, oup, reduction=32):
        """初始化 CoordinateAttention 模块。

        参数:
            inp (int): 输入通道数。
            oup (int): 输出通道数。
            reduction (int): 通道缩减比例，默认为 32。
        """
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x) -> torch.Tensor:
        """CoordinateAttention 的前向传播。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 应用坐标注意力后的输出张量。
        """
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
