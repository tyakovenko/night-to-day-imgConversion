"""
U-Net encoder-decoder for low-light to daylight image enhancement.

Architecture: 4-level U-Net with skip connections.
Input/output: 3-channel RGB, values in [0, 1].
# SOURCE: Ronneberger et al. 2015 https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample + ConvBlock (avoids checkerboard vs. transposed conv)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        )
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    4-level U-Net.

    Args:
        base_filters: Number of filters in first encoder level.
                      Doubles at each level. Default 32 (CPU-friendly).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)        # -> f
        self.enc2 = ConvBlock(f, f * 2)              # -> 2f
        self.enc3 = ConvBlock(f * 2, f * 4)          # -> 4f
        self.enc4 = ConvBlock(f * 4, f * 8)          # -> 8f
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)   # -> 16f

        # Decoder
        self.dec4 = UpBlock(f * 16, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decode with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out_conv(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
