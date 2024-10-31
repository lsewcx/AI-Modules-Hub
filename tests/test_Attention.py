import pytest
import torch
import torch.nn as nn
from ai_modules_hub.Attention import CoordinateAttention

class TestCoordinateAttention:
    def test_forward(self):
        in_channels = 64
        out_channels = 64
        reduction = 32
        model = CoordinateAttention(in_channels, out_channels, reduction)
        
        input_tensor = torch.randn(1, in_channels, 32, 32)
        output_tensor = model(input_tensor)
        
        assert output_tensor.shape == input_tensor.shape, "Output shape mismatch"

