---
language: en
license: mit
tags:
  - image-to-image
  - low-light-enhancement
  - unet
  - pytorch
  - computer-vision
  - fine-tuned
datasets:
  - tyakovenko/night-to-day-enhancement
metrics:
  - mse
---

# Night-to-Day Image Enhancement — v1-extended (TA + LOL Fine-tune)

Fine-tuned U-Net model for low-light to daylight image enhancement. Extends `best.pt` (Transient Attributes only) with additional fine-tuning on the LOL dataset for improved indoor/mixed-light generalisation.

## Model Details

| Property | Value |
|---|---|
| Architecture | U-Net, 4-level encoder-decoder |
| Parameters | 1,811,811 |
| Base filters | 16 |
| Input/Output | 3-channel RGB, values in [0, 1] |
| Loss function | MSE |
| Initialized from | `best.pt` (epoch 22) |
| Best epoch | 19 (fine-tuning) |
| Checkpoint | `best_extended.pt` |

## Training Data

**Datasets:**
- [Transient Attributes](http://transattr.cs.brown.edu/) — 102 outdoor scenes, 1,176 pairs
- [LOL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) — 485 indoor paired low/normal-light images

**HF Dataset repo:** `tyakovenko/night-to-day-enhancement`
**Combined pairs:** 1,515 train / 161 val (scene-level split)

| Source | Train | Val |
|---|---|---|
| Transient Attributes | 1,095 | 81 |
| LOL | 420 | 80 |
| **Total** | **1,515** | **161** |

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Crop size | 128×128 |
| Batch size | 8 |
| LR | 1e-5 (10× lower than original — fine-tuning) |
| Optimizer | Adam |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) |

## Evaluation Results

Evaluated on a held-out 1024×737 night/day image pair. Channel-wise MSE (float64, range [0, 1]):

| Metric | v1 (`best.pt`) | v1-extended (`best_extended.pt`) | Δ |
|---|---|---|---|
| Val MSE avg | 0.028953 | 0.027752 | **−4.1%** |
| Val MSE — R | — | 0.026945 | |
| Val MSE — G | — | 0.025522 | |
| Val MSE — B | — | 0.030788 | |

Fine-tuning on LOL improved val MSE by 4.1% and improved generalisation to non-outdoor scenes.

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model import UNet  # download model.py from this repo

# Load model
ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model", "best_extended.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = UNet(base_filters=ckpt["args"]["base_filters"])
model.load_state_dict(ckpt["model"])
model.eval()

# Inference
img = Image.open("night.jpg").convert("RGB")
to_tensor = transforms.ToTensor()
x = to_tensor(img).unsqueeze(0)  # [1, 3, H, W]

# Pad to multiple of 16 (required by U-Net with 4 MaxPool layers)
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

## Limitations

- MSE loss still produces mildly washed-out outputs for highly ambiguous dark inputs
- LOL dataset is indoor-only; fine-tuning improves indoor generalisation but may slightly reduce performance on outdoor night scenes
- See the v2 model (`tyakovenko/night-to-day-enhancement-model-v2`) for a version trained with L1 + MS-SSIM loss for improved perceptual quality
