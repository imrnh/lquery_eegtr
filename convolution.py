"""
Frequency bin convolution for a time frame.
"""
import torch
import torch.nn as nn

class FrequencyBinConvolution(nn.Module):
    def __init__(self, in_channels=26, kernel_size=(101, 1), embed_dim=512):
        super(FrequencyBinConvolution, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # depthwise: 1 kernel per channel
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(0, 0),
            groups=in_channels
        )
        # Project each channel to embed_dim separately
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, channels=26, freq_bins=101, time_steps=99)
        """
        x = self.conv(x)          # (batch, 26, 1, 99)
        x = x.permute(0, 3, 1, 2) # (batch, 99, 26, 1)
        x = self.proj(x)          # (batch, 99, 26, 512)
        return x
