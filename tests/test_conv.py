import torch

from ai_modules_hub.conv import DepthwiseSeparableConvolution


class TestDepthwiseSeparableConvolution:
    def test_forward(self):
        in_channels = 64
        out_channels = 64
        model = DepthwiseSeparableConvolution(in_channels, out_channels)
        input = torch.randn(1, in_channels, 32, 32)
        output = model(input)
        assert output.shape == input.shape, "Output shape mismatch"
