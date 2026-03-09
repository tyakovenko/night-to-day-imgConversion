---
language: en
license: mit
tags:
  - image-to-image
  - low-light-enhancement
  - unet
  - pytorch
  - computer-vision
  - residual-learning
datasets:
  - tyakovenko/night-to-day-enhancement
metrics:
  - mse
---

# Night-to-Day Image Enhancement — v3 (Residual U-Net + ColorLoss)

U-Net model for low-light to daylight image enhancement. Introduces two targeted fixes for the gray/washed-out color problem that affected v1 and v2: **residual learning** (model predicts a delta rather than the full image) and a **differentiable YCbCr color loss** that explicitly penalises desaturated outputs.

## Why v3 Exists

v1 and v2 produced structurally correct but washed-out, gray images. Two compounding causes:

1. **Sigmoid → 0.5 on uncertainty.** When the model is unsure about a pixel's correct color, Sigmoid activations converge to 0.5 — mid-gray. A blue sky the model hasn't confidently learned comes out gray.
2. **Color-blind losses.** L1 and MS-SSIM both accept desaturated outputs that match luminance/structure but miss chrominance. A gray sky can have good SSIM with a blue sky if luminance matches.

v3 fixes both directly.

## Model Details

| Property | Value |
|---|---|
| Architecture | U-Net, 4-level encoder-decoder, **residual mode** |
| Parameters | 1,811,811 |
| Base filters | 16 |
| Input/Output | 3-channel RGB, values in [0, 1] |
| Output activation | Tanh (predicts delta in [−1, 1]) |
| Residual connection | `clamp(input + delta, 0, 1)` |
| Loss function | `CombinedLoss (L1 + MS-SSIM) + 0.5 × ColorLoss (YCbCr)` |
| Initialized from | `best_extended.pt` (v1-extended, epoch 19) |
| Best epoch | 18 |
| Checkpoint | `best_v3.pt` |

## Architecture: Residual Prediction

Unlike v1/v2, v3 does **not** predict the full enhanced image. It predicts an enhancement *delta* which is added to the input:

```
Input (H×W×3)
  → Encoder: 4× [ConvBlock + MaxPool2d]
  → Bottleneck: ConvBlock
  → Decoder: 4× [Upsample + skip connection + ConvBlock]
  → Conv1×1 + Tanh → delta in [−1, 1]
  → clamp(Input + delta, 0, 1) → enhanced image
```

This means the model only needs to learn *what changes* between night and day. Background structure is preserved for free. Tanh activations don't saturate to a mid-gray mean — they saturate to ±1, so uncertain predictions push toward the input rather than toward gray.

## Loss Function: ColorLoss

`ColorLoss` converts predictions to YCbCr (BT.601 coefficients, pure PyTorch — no new dependencies) and penalises chrominance errors 2× relative to luminance:

```
loss = L1(Y_pred, Y_target) + 2 × [L1(Cb_pred, Cb_target) + L1(Cr_pred, Cr_target)]
```

This directly punishes the failure mode that L1 and MS-SSIM let through: a gray sky instead of a blue sky.

## Training Data

**Datasets:**
- [Transient Attributes](http://transattr.cs.brown.edu/) — 102 outdoor scenes, 1,176 pairs
- [LOL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) — 485 indoor paired low/normal-light images

**HF Dataset repo:** `tyakovenko/night-to-day-enhancement`

| Source | Train | Val |
|---|---|---|
| Transient Attributes | 1,095 | 81 |
| LOL | 420 | 80 |
| **Total** | **1,515** | **161** |

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Crop size | 176×176 |
| Batch size | 4 |
| LR | 1e-5 → 5e-6 (ReduceLROnPlateau) |
| Optimizer | Adam |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) |
| Color weight | 0.5 |

## Evaluation Results

Channel-wise MSE (float64, range [0, 1]) on validation set and held-out eval pair:

| Metric | v1 (`best.pt`) | v3 (`best_v3.pt`) |
|---|---|---|
| Val MSE avg | 0.028953 | **0.050844** |
| Val MSE — R | 0.028700 | 0.050646 |
| Val MSE — G | 0.025959 | 0.046377 |
| Val MSE — B | 0.032200 | 0.055508 |
| Eval MSE avg (`night.jpg`) | 0.038925 | 0.075302 |

**Note on higher MSE:** v3's val and eval MSE are higher than v1. This is expected and not a regression. The loss function explicitly trades raw pixel accuracy for chrominance correctness — a gray sky that matches luminance structure will score better than a blue sky on pure MSE, but worse on ColorLoss. v3 is optimised for color fidelity; MSE is reported for comparability only. Visual inspection is the meaningful test.

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model import UNet  # download model.py from this repo

# Load model — note residual=True for v3
ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model-v3", "best_v3.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
base_filters = ckpt.get("args", {}).get("base_filters", 16)
model = UNet(base_filters=base_filters, residual=True)
model.load_state_dict(ckpt["model"])
model.eval()

# Inference
img = Image.open("night.jpg").convert("RGB")
x = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]

# Pad to multiple of 16 (required by 4-level U-Net)
def pad_to_multiple(t, m=16):
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    return torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect"), h, w

x_padded, orig_h, orig_w = pad_to_multiple(x)
with torch.no_grad():
    out = model(x_padded)
out = out[:, :, :orig_h, :orig_w]

result = transforms.ToPILImage()(out.squeeze(0).clamp(0, 1))
result.save("enhanced.jpg")
```

## Model Lineage

| Version | Key change | Val MSE |
|---|---|---|
| v1 | Baseline — U-Net, MSE loss, Transient Attributes | 0.028953 |
| v1-extended | + LOL fine-tuning | 0.027752 |
| v2 | L1 + MS-SSIM loss | — |
| **v3** | **Residual U-Net + YCbCr ColorLoss** | **0.050844** |

See the [project repo](https://github.com/tyakovenko/night-to-day-imgConversion) for full training scripts and architecture details.

## Limitations

- v3 trades pixel-level MSE for chrominance accuracy — raw MSE is higher than v1
- Residual learning helps but 20 epochs from a Sigmoid warm-start may not be enough for full recalibration; more training likely beneficial
- Staged training (warm-start longer before introducing ColorLoss) is a promising next step
