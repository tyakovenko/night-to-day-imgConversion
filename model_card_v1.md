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

# Night-to-Day Image Enhancement — v1 (Transient Attributes)

U-Net model trained to enhance low-light and night-time images to match daylight appearance.

## Model Details

| Property | Value |
|---|---|
| Architecture | U-Net, 4-level encoder-decoder |
| Parameters | 1,811,811 |
| Base filters | 16 |
| Input/Output | 3-channel RGB, values in [0, 1] |
| Loss function | MSE |
| Best epoch | 22 |
| Checkpoint | `best.pt` |

## Training Data

**Dataset:** [Transient Attributes](http://transattr.cs.brown.edu/) — 102 outdoor scenes, time-of-day captures.
**HF Dataset repo:** `tyakovenko/night-to-day-enhancement`
**Pairs:** 1,010 train / 166 val (scene-level split, 1,176 total)

Pair selection: day targets with `daylight > 0.8`; low-light inputs with `night > 0.5` OR (`dark > 0.5` AND `daylight < 0.4`) OR (`dawndusk > 0.5` AND `daylight < 0.4`).

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Crop size | 128×128 |
| Batch size | 8 |
| Initial LR | 1e-4 → 5e-5 → 2.5e-5 |
| Optimizer | Adam |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) |

## Evaluation Results

Evaluated on a held-out 1024×737 night/day image pair. Channel-wise MSE (float64, range [0, 1]):

| Metric | Value |
|---|---|
| Val MSE avg (epoch 22) | 0.028953 |
| Final eval MSE — R | 0.037988 |
| Final eval MSE — G | 0.035524 |
| Final eval MSE — B | 0.043262 |
| **Final eval MSE avg** | **0.038925** |

Blue channel is consistently the hardest; green learns fastest.

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model import UNet  # download model.py from this repo

# Load model
ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model", "best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = UNet(base_filters=ckpt["args"]["base_filters"])
model.load_state_dict(ckpt["model"])
model.eval()

# Inference
img = Image.open("night.jpg").convert("RGB")
to_tensor = transforms.ToTensor()
x = to_tensor(img).unsqueeze(0)  # [1, 3, H, W]

# Pad to multiple of 16
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

- Trained only on outdoor scenes (Transient Attributes dataset)
- At 1.8M parameters, capacity is limited — may produce washed-out results on very dark inputs
- Loss function (MSE) tends toward regression-to-mean for highly ambiguous inputs
- See `best_extended.pt` in this repo for a fine-tuned version with better generalisation
