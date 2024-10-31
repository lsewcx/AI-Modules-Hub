import torch
from ai_modules_hub.Attention import CoordinateAttention


class TestCoordinateAttention:
    def test_forward(self):
        in_channels = 64
        out_channels = 64
        reduction = 32
        model = CoordinateAttention(in_channels, out_channels, reduction)
        input = torch.randn(1, in_channels, 32, 32)
        output = model(input)
        assert output.shape == input.shape, "Output shape mismatch"
