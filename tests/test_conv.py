import torch

from ai_modules_hub.conv import DepthwiseSeparableConvolution

class TestDepthwiseSeparableConvolution:
    def test_forward(self):
        in_channels = 64
        out_channels = 64
        model = DepthwiseSeparableConvolution(in_channels, out_channels)
        
        input_tensor = torch.randn(1, in_channels, 32, 32)
        output_tensor = model(input_tensor)
        
        assert output_tensor.shape == input_tensor.shape, "Output shape mismatch"