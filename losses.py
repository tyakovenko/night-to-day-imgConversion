"""
Loss functions for low-light image enhancement.

CombinedLoss    — L1 + MS-SSIM (pixel fidelity + perceptual quality)
ColorLoss       — YCbCr chrominance loss (penalises desaturated/gray outputs)
WeightedL1Loss  — input-luminance-weighted L1 (suppresses lamp gradients)
LogL1Loss       — log-domain L1 (auto down-weights bright pixels)
V4Loss          — WeightedL1 + LogL1 + MS-SSIM + ColorLoss (with staged warmup)
PerceptualLoss  — VGG-16 feature matching (relu2_2 + relu3_3)
EnhancementLoss — CombinedLoss + PerceptualLoss (full training loss)

Reference: Wang et al., "Multi-Scale Structural Similarity for Image Quality Assessment"
           Loss weighting: 0.84 * (1 - MS-SSIM) + 0.16 * L1 (from the paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim  # pip install pytorch-msssim


class CombinedLoss(nn.Module):
    """
    Pixel-level loss: 0.84 * (1 - MS-SSIM) + 0.16 * L1.

    Wang et al. weighting. MS-SSIM optimises luminance, contrast, and structure
    at multiple scales; L1 anchors absolute pixel values and avoids the
    regression-to-mean saturation that pure SSIM can produce.

    Args:
        data_range: max value of inputs (1.0 for float tensors in [0,1])
    """

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        # ms_ssim returns a scalar in [0, 1]; higher = more similar
        ssim_val = ms_ssim(pred, target, data_range=self.data_range, size_average=True)
        return 0.16 * l1 + 0.84 * (1.0 - ssim_val)


class ColorLoss(nn.Module):
    """
    Penalise chrominance (Cb, Cr) mismatch more than luminance (Y).

    Uses BT.601 RGB→YCbCr coefficients — fully differentiable, no new deps.
    Chrominance errors are weighted 2× relative to luminance, because L1 and
    MS-SSIM both accept desaturated (gray) outputs that match structure but
    miss color temperature. This loss explicitly punishes that failure mode.

    # SOURCE: BT.601 coefficients — https://www.itu.int/rec/R-REC-BT.601/
    """

    @staticmethod
    def _rgb_to_ycbcr(x: torch.Tensor):
        """Convert [B,3,H,W] float tensor in [0,1] to Y, Cb, Cr components."""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  =  0.299   * r + 0.587   * g + 0.114   * b
        cb = -0.16875 * r - 0.33126 * g + 0.5     * b + 0.5
        cr =  0.5     * r - 0.41869 * g - 0.08131 * b + 0.5
        return y, cb, cr

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_p,  cb_p,  cr_p  = self._rgb_to_ycbcr(pred)
        y_t,  cb_t,  cr_t  = self._rgb_to_ycbcr(target)
        lum   = F.l1_loss(y_p,  y_t)
        chrom = F.l1_loss(cb_p, cb_t) + F.l1_loss(cr_p, cr_t)
        return lum + 2.0 * chrom


class WeightedL1Loss(nn.Module):
    """
    Input-luminance-weighted L1 loss.

    Bright pixels (streetlamps, windows) contribute almost no gradient because
    their weight approaches zero. Dark ambient pixels dominate the loss signal,
    steering the model to correctly illuminate dark regions instead of
    preserving lamp intensity.

        weight = 1 - clamp(Y_night, 0, 1)

    where Y_night is the BT.601 luminance of the *input* low-light image
    (not pred or target). The caller must pass the original input as input_img.

    # SOURCE: BT.601 luminance — https://www.itu.int/rec/R-REC-BT.601/
    """

    @staticmethod
    def _luminance(x: torch.Tensor) -> torch.Tensor:
        """BT.601 luminance from [B,3,H,W] RGB tensor in [0,1]."""
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_img: torch.Tensor,
    ) -> torch.Tensor:
        weight = 1.0 - torch.clamp(self._luminance(input_img), 0.0, 1.0)
        return (weight * torch.abs(pred - target)).mean()


class LogL1Loss(nn.Module):
    """
    L1 loss in log domain: L1(log(pred + eps), log(target + eps)).

    Compresses the gradient from bright pixels: the distance between 0.90 and
    0.95 is tiny in log space, while the distance between 0.05 and 0.10 is
    large. Automatically down-weights lamp intensity without explicit masking.

    Args:
        eps: small constant to avoid log(0). Default 1e-3.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(
            torch.log(pred + self.eps),
            torch.log(target + self.eps),
        )


class V4Loss(nn.Module):
    """
    v4 training loss.

    Components:
      - WeightedL1:  input-luminance-weighted L1 (suppresses lamp gradients)
      - LogL1:       log-domain L1 (auto down-weights bright pixels)
      - MS-SSIM:     structural similarity
      - ColorLoss:   YCbCr chrominance (with epoch-based warmup via set_color_scale)

    The caller must pass the original low-light input as the third argument:
        loss = criterion(pred, target, input_img)

    ColorLoss is gated by _color_scale (0 → disabled, 1 → full weight).
    Call set_color_scale(scale) at the start of each epoch to implement
    the warmup schedule.

    Args:
        w_weighted_l1: weight for WeightedL1 term
        w_log_l1:      weight for LogL1 term
        w_ssim:        weight for (1 - MS-SSIM) term
        color_weight:  final weight for ColorLoss
        data_range:    max value of inputs (1.0 for [0,1] tensors)
    """

    def __init__(
        self,
        w_weighted_l1: float = 0.16,
        w_log_l1: float = 0.16,
        w_ssim: float = 0.68,
        color_weight: float = 0.5,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.weighted_l1  = WeightedL1Loss()
        self.log_l1       = LogL1Loss()
        self.color        = ColorLoss()
        self.w_wl1        = w_weighted_l1
        self.w_ll1        = w_log_l1
        self.w_ssim       = w_ssim
        self.color_weight = color_weight
        self.data_range   = data_range
        self._color_scale = 0.0  # updated each epoch by the trainer

    def set_color_scale(self, scale: float) -> None:
        """Set the current ColorLoss scale (0 = disabled, 1 = full weight)."""
        self._color_scale = float(max(0.0, min(1.0, scale)))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input_img: torch.Tensor,
    ) -> torch.Tensor:
        ssim_val = ms_ssim(pred, target, data_range=self.data_range, size_average=True)
        loss = (
            self.w_wl1  * self.weighted_l1(pred, target, input_img)
            + self.w_ll1 * self.log_l1(pred, target)
            + self.w_ssim * (1.0 - ssim_val)
        )
        if self._color_scale > 0.0:
            loss = loss + self._color_scale * self.color_weight * self.color(pred, target)
        return loss


class PerceptualLoss(nn.Module):
    """
    VGG-16 feature loss using activations from relu2_2 (layer 9) and relu3_3 (layer 16).

    The frozen VGG extracts semantic features. MSE in feature space penalises
    perceptually meaningful differences (textures, edges, structure) rather than
    per-pixel differences, which produces sharper and more realistic outputs.

    # SOURCE: https://arxiv.org/abs/1603.08155 (Johnson et al., perceptual losses)
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())

        # relu2_2: index 9 (inclusive), relu3_3: index 16 (inclusive)
        self.slice1 = nn.Sequential(*features[:10])   # up to relu2_2
        self.slice2 = nn.Sequential(*features[10:17]) # relu2_2 → relu3_3

        # Freeze — we never train VGG
        for p in self.parameters():
            p.requires_grad = False

        # VGG expects ImageNet-normalised input
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise [0,1] RGB tensor to ImageNet statistics."""
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n   = self._normalize(pred)
        target_n = self._normalize(target)

        feat1_p = self.slice1(pred_n)
        feat1_t = self.slice1(target_n)
        loss1   = F.mse_loss(feat1_p, feat1_t)

        feat2_p = self.slice2(feat1_p)
        feat2_t = self.slice2(feat1_t)
        loss2   = F.mse_loss(feat2_p, feat2_t)

        return loss1 + loss2


class EnhancementLoss(nn.Module):
    """
    Full training loss: CombinedLoss (pixel) + perceptual_weight * PerceptualLoss (VGG).

    Default perceptual_weight=0.1 balances pixel fidelity with perceptual quality.
    Increase to 0.2–0.5 for sharper textures; decrease if colour accuracy suffers.

    Args:
        perceptual_weight: scale factor for VGG feature loss
        data_range: max value of inputs (1.0 for [0,1] float tensors)
    """

    def __init__(self, perceptual_weight: float = 0.1, data_range: float = 1.0):
        super().__init__()
        self.pixel_loss  = CombinedLoss(data_range=data_range)
        self.perceptual  = PerceptualLoss()
        self.pw          = perceptual_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.pixel_loss(pred, target)
        if self.pw > 0:
            loss = loss + self.pw * self.perceptual(pred, target)
        return loss
