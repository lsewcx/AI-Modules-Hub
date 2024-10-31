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
    论文地址:https://arxiv.org/pdf/2103.02907
    github源码地址:https://github.com/houqb/CoordAttention/blob/main/coordatt.py
    """

    def __init__(self, inp, oup, reduction=32):
        """
        初始化 CoordinateAttention 模块。

        参数:
        inp (int): 输入通道数。
        oup (int): 输出通道数。
        reduction (int): 通道缩减比例，默认为32。
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


if __name__ == "__main__":
    in_channels = 64
    out_channels = 64
    reduction = 32
    x = torch.randn(1, in_channels, 64, 64)
    model = CoordinateAttention(in_channels, out_channels, reduction)
    out = model(x)
    print(out.shape)
    assert out.shape == x.shape, "CoordinateAttention shape error"
