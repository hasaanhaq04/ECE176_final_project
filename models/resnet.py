"""
ResNet-50 implemented from scratch in PyTorch.

Supports four attention variants injected into every Bottleneck block:
  'none'    — vanilla ResNet-50
  'channel' — ChannelAttention only
  'spatial' — SpatialAttention only
  'cbam'    — full CBAM (channel then spatial)

CIFAR-100 adaptation: the standard ImageNet stem (7×7 conv, stride-2, maxpool)
is replaced by a single 3×3 conv (stride-1) to preserve spatial resolution for
32×32 inputs.
"""

import torch
import torch.nn as nn
from .cbam import ChannelAttention, SpatialAttention, CBAM

ATTENTION_TYPES = ('none', 'channel', 'spatial', 'cbam')


def _make_attention(attention_type: str, channels: int, reduction: int) -> nn.Module | None:
    """Return the appropriate attention module or None."""
    if attention_type == 'channel':
        return ChannelAttention(channels, reduction)
    elif attention_type == 'spatial':
        return SpatialAttention(kernel_size=7)
    elif attention_type == 'cbam':
        return CBAM(channels, reduction, spatial_kernel=7)
    return None


class Bottleneck(nn.Module):
    """
    ResNet Bottleneck block: 1×1 → 3×3 → 1×1 convolutions.

    Optional attention module is applied to the residual (conv) branch output
    before adding the skip connection, matching the CBAM paper placement.

    Args:
        in_channels:    Channels coming in.
        planes:         Base channel width (output = planes * expansion).
        stride:         Stride for the 3×3 conv (controls downsampling).
        downsample:     Optional projection shortcut (1×1 conv + BN).
        attention_type: One of 'none', 'channel', 'spatial', 'cbam'.
        reduction:      Channel attention reduction ratio.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        attention_type: str = 'none',
        reduction: int = 16,
    ):
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.attention = _make_attention(attention_type, out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Apply attention to the residual branch before the skip add
        if self.attention is not None:
            out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet50(nn.Module):
    """
    ResNet-50 for CIFAR-100 (32×32 input).

    The ImageNet stem (7×7 conv + maxpool) is replaced by a single
    3×3 conv (stride 1) so that the spatial resolution is preserved for
    the small 32×32 images.

    Args:
        num_classes:    Number of output classes (100 for CIFAR-100).
        attention_type: Attention variant for all Bottleneck blocks.
        reduction:      Channel reduction ratio used by ChannelAttention / CBAM.
    """

    def __init__(
        self,
        num_classes: int = 100,
        attention_type: str = 'none',
        reduction: int = 16,
    ):
        super().__init__()
        assert attention_type in ATTENTION_TYPES, f"Unknown attention_type '{attention_type}'"

        self.attention_type = attention_type
        self.reduction      = reduction
        self.in_channels    = 64

        # CIFAR stem: 3×3 conv, stride 1 (no maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Four residual stages — [3, 4, 6, 3] blocks, planes [64, 128, 256, 512]
        self.layer1 = self._make_layer(64,  3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        out_channels = planes * Bottleneck.expansion

        # Projection shortcut when spatial size or channel count changes
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            Bottleneck(
                self.in_channels, planes,
                stride=stride,
                downsample=downsample,
                attention_type=self.attention_type,
                reduction=self.reduction,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(
                Bottleneck(
                    self.in_channels, planes,
                    attention_type=self.attention_type,
                    reduction=self.reduction,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(
    variant: str,
    num_classes: int = 100,
    reduction: int = 16,
) -> ResNet50:
    """
    Factory function for the four experimental variants.

    Args:
        variant:     One of 'baseline', 'channel', 'spatial', 'cbam'.
        num_classes: Output class count.
        reduction:   Channel attention reduction ratio.

    Returns:
        ResNet50 instance with the requested attention configuration.
    """
    variant_map = {
        'baseline': 'none',
        'channel':  'channel',
        'spatial':  'spatial',
        'cbam':     'cbam',
    }
    assert variant in variant_map, f"variant must be one of {list(variant_map)}"
    return ResNet50(num_classes=num_classes, attention_type=variant_map[variant], reduction=reduction)
