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


class GlobalContextEncoder(nn.Module):
    """
    Encode per-channel global statistics of the full input image into a channel
    embedding that is broadcast-added to the U-Net bottleneck.

    Input:  [B, 9] — (mean_R, mean_G, mean_B, std_R, std_G, std_B, p10_R, p10_G, p10_B)
    Output: [B, out_channels, 1, 1] — broadcast-ready for addition to bottleneck

    This lets the model distinguish a pixel that is bright because of a streetlamp
    (locally isolated bright spot in an otherwise dark scene) from ambient daylight
    (globally high luminance). The crop-level encoder cannot see this distinction.

    Args:
        out_channels: must match the bottleneck channel count (base_filters * 16)
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_channels),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        # stats: [B, 9] -> [B, out_channels] -> [B, out_channels, 1, 1]
        return self.net(stats).unsqueeze(-1).unsqueeze(-1)


class UNet(nn.Module):
    """
    4-level U-Net.

    Args:
        base_filters: Number of filters in first encoder level.
                      Doubles at each level. Default 32 (CPU-friendly).
        residual: If True, use residual learning — model predicts a Tanh delta
                  which is added to the input and clamped to [0, 1]. This lets
                  the model focus on *what changes* rather than reconstructing
                  the full image from scratch, which fixes gray/washed-out outputs.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_filters: int = 32, residual: bool = False,
                 use_global_context: bool = False):
        super().__init__()
        f = base_filters
        self.residual = residual
        self.use_global_context = use_global_context

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)        # -> f
        self.enc2 = ConvBlock(f, f * 2)              # -> 2f
        self.enc3 = ConvBlock(f * 2, f * 4)          # -> 4f
        self.enc4 = ConvBlock(f * 4, f * 8)          # -> 8f
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)   # -> 16f

        # Optional global context injection at bottleneck.
        # Encodes scene-level stats (mean/std/p10 per channel) into a
        # (f*16)-dim embedding added to every spatial position in the bottleneck.
        if use_global_context:
            self.global_ctx = GlobalContextEncoder(out_channels=f * 16)

        # Decoder
        self.dec4 = UpBlock(f * 16, f * 8, f * 8)
        self.dec3 = UpBlock(f * 8, f * 4, f * 4)
        self.dec2 = UpBlock(f * 4, f * 2, f * 2)
        self.dec1 = UpBlock(f * 2, f, f)

        # Output head — Tanh for residual mode (delta in [-1,1]), Sigmoid otherwise
        activation = nn.Tanh() if residual else nn.Sigmoid()
        self.out_conv = nn.Sequential(
            nn.Conv2d(f, out_channels, kernel_size=1),
            activation,
        )

    def forward(self, x, global_stats=None):
        """
        Args:
            x:            [B, 3, H, W] low-light input
            global_stats: [B, 9] per-channel (mean, std, p10) of the full input
                          image. Required when use_global_context=True.
        """
        x_input = x  # saved for residual addition

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Inject global context: broadcast [B, C, 1, 1] across spatial dims
        if self.use_global_context and global_stats is not None:
            b = b + self.global_ctx(global_stats)

        # Decode with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.out_conv(d1)

        if self.residual:
            # Add predicted delta to input; clamp keeps output in valid [0,1] range
            out = torch.clamp(x_input + out, 0.0, 1.0)

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
