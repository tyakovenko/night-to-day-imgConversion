---
language: en
license: mit
tags:
  - image-to-image
  - low-light-enhancement
  - unet
  - pytorch
  - computer-vision
datasets:
  - tyakovenko/night-to-day-enhancement
metrics:
  - mse
---

# Night-to-Day Image Enhancement — Model Repo

U-Net models trained to enhance low-light and night-time images to match daylight appearance.

This repo contains two checkpoints:

| Checkpoint | Training data | Best val MSE | Notes |
|---|---|---|---|
| `best.pt` | Transient Attributes only | 0.028953 | v1 baseline |
| `best_extended.pt` | TA + LOL fine-tune | **0.027752** | Recommended |

---

## Architecture

- **Model:** U-Net, 4-level encoder-decoder with skip connections
- **Parameters:** 1,811,811
- **Base filters:** 16
- **Input/Output:** 3-channel RGB, float32 in [0, 1]
- **Output activation:** Sigmoid

---

## `best.pt` — v1 (Transient Attributes only)

### Training Data
- **Dataset:** [Transient Attributes](http://transattr.cs.brown.edu/) — 102 outdoor scenes, time-of-day captures
- **HF Dataset:** `tyakovenko/night-to-day-enhancement`
- **Split:** 1,010 train / 166 val (scene-level)

### Training Config

| Hyperparameter | Value |
|---|---|
| Loss | MSE |
| Crop size | 128×128 |
| Batch size | 8 |
| LR schedule | 1e-4 → 5e-5 → 2.5e-5 (ReduceLROnPlateau) |
| Best epoch | 22 |

### Results

| Metric | Value |
|---|---|
| Val MSE avg | 0.028953 |
| Final eval MSE — R | 0.037988 |
| Final eval MSE — G | 0.035524 |
| Final eval MSE — B | 0.043262 |
| **Final eval MSE avg** | **0.038925** |

---

## `best_extended.pt` — v1-extended (TA + LOL fine-tune)

Fine-tuned from `best.pt`. Adds the LOL indoor dataset to improve generalisation to non-outdoor scenes.

### Training Data
- **Datasets:** Transient Attributes + [LOL](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) (485 indoor paired images)
- **Combined split:** 1,515 train / 161 val

| Source | Train | Val |
|---|---|---|
| Transient Attributes | 1,095 | 81 |
| LOL | 420 | 80 |
| **Total** | **1,515** | **161** |

### Training Config

| Hyperparameter | Value |
|---|---|
| Loss | MSE |
| Initialized from | `best.pt` (epoch 22) |
| Crop size | 128×128 |
| Batch size | 8 |
| LR | 1e-5 (fine-tuning, 10× lower) |
| Best epoch | 19 |

### Results

| Metric | `best.pt` | `best_extended.pt` | Δ |
|---|---|---|---|
| Val MSE avg | 0.028953 | **0.027752** | −4.1% |
| Val MSE — R | — | 0.026945 | |
| Val MSE — G | — | 0.025522 | |
| Val MSE — B | — | 0.030788 | |

---

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model import UNet  # available in this repo

REPO = "tyakovenko/night-to-day-enhancement-model"

# Choose checkpoint: "best.pt" or "best_extended.pt"
ckpt_path = hf_hub_download(REPO, "best_extended.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = UNet(base_filters=ckpt["args"]["base_filters"])
model.load_state_dict(ckpt["model"])
model.eval()

def pad_to_multiple(t, m=16):
    """Pad spatial dims to a multiple of m (required by 4-level U-Net)."""
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    return torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect"), h, w

img = Image.open("night.jpg").convert("RGB")
x = transforms.ToTensor()(img).unsqueeze(0)
x_padded, orig_h, orig_w = pad_to_multiple(x)

with torch.no_grad():
    out = model(x_padded)

out = out[:, :, :orig_h, :orig_w].clamp(0, 1)
transforms.ToPILImage()(out.squeeze(0)).save("enhanced.jpg")
```

---

## Limitations

- Both checkpoints use MSE loss, which can produce mildly washed-out results on very dark inputs
- `best.pt` trained on outdoor scenes only — may underperform on indoor images
- `best_extended.pt` improves indoor generalisation but the LOL weighting is not tuned; outdoor performance is largely preserved
- See `tyakovenko/night-to-day-enhancement-model-v2` for a version trained with L1 + MS-SSIM loss for improved perceptual quality
