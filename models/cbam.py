"""
CBAM: Convolutional Block Attention Module
Woo et al., ECCV 2018 — implemented from scratch.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Squeezes spatial dimensions via global avg + max pooling, then uses a
    shared two-layer MLP to produce per-channel attention weights.

    Args:
        channels:  Number of input channels.
        reduction: Hidden-layer reduction ratio for the MLP (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Global average pool and global max pool → (B, C)
        avg = x.mean(dim=[2, 3])
        mx  = x.amax(dim=[2, 3])

        # Shared MLP (same weights applied to both)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)

        # Recalibrate channels
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Compresses channel dimension via avg + max across channels, concatenates
    them, then applies a single convolution to produce a spatial attention map.

    Args:
        kernel_size: Convolution kernel size (default 7, as in the paper).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, H, W) channel-wise stats
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)

        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(combined))         # (B, 1, H, W)

        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies ChannelAttention then SpatialAttention to refine
    an intermediate feature map.

    Args:
        channels:       Number of input channels.
        reduction:      Channel attention MLP reduction ratio (default 16).
        spatial_kernel: Spatial attention conv kernel size (default 7).
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
