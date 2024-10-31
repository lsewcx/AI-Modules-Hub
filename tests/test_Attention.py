import torch
from ai_modules_hub.Attention import CoordinateAttention, MobileViTAttention


class TestCoordinateAttention:
    def test_forward(self):
        in_channels = 64
        out_channels = 64
        reduction = 32
        model = CoordinateAttention(in_channels, out_channels, reduction)
        input = torch.randn(1, in_channels, 32, 32)
        output = model(input)
        assert output.shape == input.shape, "Output shape mismatch"


class TestMobileViTAttention:
    def test_forward(self):
        model = MobileViTAttention(image_size=(256, 256), dims=[144, 192, 240], channels=[16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640], num_classes=1000)
        input_tensor = torch.randn(1, 3, 256, 256)
        output_tensor = model(input_tensor)
        assert output_tensor.shape == (1, 1000), "Output shape mismatch"
