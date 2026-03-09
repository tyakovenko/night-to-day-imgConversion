"""
Upload best_v2.pt to a new HF Hub model repo: tyakovenko/night-to-day-enhancement-model-v2

Run automatically after train_v2.py completes. Uploads:
  - checkpoints/best_v2.pt   (model weights)
  - model.py                 (architecture)
  - losses.py                (loss functions used in training)
  - README.md                (model card)
"""

import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo

REPO_ID   = "tyakovenko/night-to-day-enhancement-model-v2"
CKPT_PATH = Path("checkpoints/best_v2.pt")


def read_best_metrics() -> dict:
    """Pull best val MSE figures from experiment_log_v2.csv."""
    log = Path("experiment_log_v2.csv")
    if not log.exists():
        return {}
    import csv
    best_row = None
    best_mse = float("inf")
    with open(log) as f:
        for row in csv.DictReader(f):
            try:
                mse = float(row["val_mse_avg"])
            except (ValueError, TypeError):
                continue  # skip duplicate header rows
            if mse < best_mse:
                best_mse = mse
                best_row = row
    return best_row or {}


def make_model_card(metrics: dict) -> str:
    epoch    = metrics.get("epoch",       "?")
    mse_avg  = metrics.get("val_mse_avg", "?")
    mse_r    = metrics.get("val_mse_R",   "?")
    mse_g    = metrics.get("val_mse_G",   "?")
    mse_b    = metrics.get("val_mse_B",   "?")

    return f"""\
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

# Night-to-Day Image Enhancement — v2 (L1 + MS-SSIM)

U-Net model trained to enhance low-light and night-time images to match daylight appearance.
This is the v2 checkpoint, trained with a composite **L1 + MS-SSIM** loss instead of MSE.

**Previous checkpoints (MSE-trained):** `tyakovenko/night-to-day-enhancement-model`

## Why L1 + MS-SSIM?

MSE loss minimises expected pixel error, which causes two well-known failure modes:

- **Washed-out / desaturated colours** — MSE averages over all plausible outputs for ambiguous inputs
- **Blurry texture** — MSE treats every pixel independently and cannot penalise spatially incoherent predictions

The v2 loss combines:
- **L1 (MAE)** — less susceptible to outliers than MSE, produces sharper absolute values
- **MS-SSIM** — directly optimises luminance, contrast, and structure at multiple scales

Loss weighting (Wang et al.): `0.84 * (1 - MS-SSIM) + 0.16 * L1`

## Model Details

| Property | Value |
|---|---|
| Architecture | U-Net, 4-level encoder-decoder |
| Parameters | 1,811,811 |
| Base filters | 16 |
| Input/Output | 3-channel RGB, float32 in [0, 1] |
| Loss | L1 + MS-SSIM (Wang et al. weighting) |
| Initialized from | `best_extended.pt` (TA+LOL, val MSE 0.027752) |
| Best epoch | {epoch} |

## Training Data

- [Transient Attributes](http://transattr.cs.brown.edu/) — 102 outdoor scenes
- [LOL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) — 485 indoor paired images
- **HF Dataset repo:** `tyakovenko/night-to-day-enhancement`
- **Split:** 1,515 train / 161 val (scene-level)

## Training Config

| Hyperparameter | Value |
|---|---|
| Crop size | 176×176 |
| Batch size | 4 |
| LR | 1e-5 |
| Optimizer | Adam |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) |

## Evaluation Results

Channel-wise MSE (float64, values in [0, 1]):

| Metric | v1-extended (MSE) | **v2 (L1+MS-SSIM)** |
|---|---|---|
| Val MSE avg | 0.027752 | **{mse_avg}** |
| Val MSE — R | 0.026945 | {mse_r} |
| Val MSE — G | 0.025522 | {mse_g} |
| Val MSE — B | 0.030788 | {mse_b} |

## Usage

```python
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model import UNet  # available in this repo

ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model-v2", "best_v2.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = UNet(base_filters=ckpt["args"]["base_filters"])
model.load_state_dict(ckpt["model"])
model.eval()

def pad_to_multiple(t, m=16):
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
"""


def main():
    if not CKPT_PATH.exists():
        print(f"ERROR: {CKPT_PATH} not found — training may not have completed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nUploading v2 model to {REPO_ID} ...")

    # Create repo (no-op if already exists)
    create_repo(REPO_ID, repo_type="model", exist_ok=True)

    metrics = read_best_metrics()
    print(f"Best checkpoint: epoch {metrics.get('epoch','?')}, "
          f"val MSE avg {metrics.get('val_mse_avg','?')}")

    # Write model card to a temp file
    card_path = Path("_model_card_v2_tmp.md")
    card_path.write_text(make_model_card(metrics))

    files = [
        (str(CKPT_PATH),  "best_v2.pt",  "Upload best_v2.pt weights"),
        ("model.py",      "model.py",    "Upload model architecture"),
        ("losses.py",     "losses.py",   "Upload loss functions"),
        (str(card_path),  "README.md",   "Upload model card"),
    ]

    for local, remote, msg in files:
        if not Path(local).exists():
            print(f"  WARNING: {local} not found, skipping.")
            continue
        print(f"  {msg} ...")
        upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO_ID,
            commit_message=msg,
        )
        print(f"  Done.")

    card_path.unlink(missing_ok=True)

    print(f"\nUpload complete.")
    print(f"  Repo: https://huggingface.co/{REPO_ID}")
    print(f"  Files: best_v2.pt, model.py, losses.py, README.md")


if __name__ == "__main__":
    main()
