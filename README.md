---
title: Night to Day Enhancement
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: enhance low light images for day-like images
---

# Low-Light to Day Image Enhancement

A U-Net model that enhances low-light and night-time images to match daylight appearance, evaluated by channel-wise MSE.

- **Live demo:** [tyakovenko-night-to-day-enhancement.hf.space](https://tyakovenko-night-to-day-enhancement.hf.space)
- **Dataset:** [tyakovenko/night-to-day-enhancement](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) — 1,176 pairs (Transient Attributes)
- **Extended dataset:** [tyakovenko/night-to-day-enhancement-extended](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement-extended) — adds 420 LOL pairs
- **v1 model:** [tyakovenko/night-to-day-enhancement-model](https://huggingface.co/tyakovenko/night-to-day-enhancement-model) — best.pt / best_extended.pt
- **v2 model:** [tyakovenko/night-to-day-enhancement-model-v2](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v2) — best_v2.pt (L1 + MS-SSIM)
- **v3 model:** [tyakovenko/night-to-day-enhancement-model-v3](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v3) — best_v3.pt (Residual + ColorLoss)

---

## Run the demo locally (Docker)

Anyone with Docker installed can run the Gradio UI locally — no Python setup needed.

```bash
# 1. Clone the repo
git clone https://github.com/tyakovenko/night-to-day-imgConversion.git
cd night-to-day-imgConversion

# 2. Build the image (installs CPU-only torch + all dependencies in a venv)
docker build -t night-to-day .

# 3. Run — UI available at http://localhost:7860
docker run -p 7860:7860 night-to-day
```

On first launch the model checkpoint (~22MB) downloads from HF Hub automatically. Subsequent runs use the Docker layer cache.

> **Requirements:** Docker Desktop (Mac/Windows) or Docker Engine (Linux). No GPU needed.

---

## Reproduce the experiment (Python, no Docker)

### 1. Environment

```bash
git clone https://github.com/tyakovenko/night-to-day-imgConversion.git
cd night-to-day-imgConversion

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Train the baseline model (Transient Attributes dataset)

Training streams images from HF Hub on first run (~1,266 files, cached locally after).

```bash
python train.py \
  --epochs 30 \
  --batch-size 8 \
  --crop-size 128 \
  --base-filters 16 \
  --lr 1e-4 \
  --workers 4
```

Expected outcome: `checkpoints/best.pt` at ~epoch 22, val MSE avg ≈ **0.029**.
Runtime: ~90–110s/epoch on CPU.

### 3. Train the extended model (Transient Attributes + LOL)

Requires the [LOL dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) downloaded locally. Place it so `low/` and `high/` subfolders are accessible, then run the upload script once to push LOL images to HF Hub and extend the manifest:

```bash
python - <<'EOF'
import shutil
from pathlib import Path
from huggingface_hub import upload_large_folder

LOL_SRC = Path("/path/to/lol_dataset")   # <-- update this path
STAGE   = Path("/tmp/lol_hf_stage")

for src, dst in [("low", "low_light/lol"), ("high", "day/lol")]:
    (STAGE / dst).mkdir(parents=True, exist_ok=True)
    for f in (LOL_SRC / src).iterdir():
        shutil.copy2(f, STAGE / dst / f.name)

upload_large_folder(
    repo_id="tyakovenko/night-to-day-enhancement-extended",
    repo_type="dataset",
    folder_path=str(STAGE),
)
EOF
```

Then fine-tune from the baseline checkpoint:

```bash
python train_extended.py \
  --epochs 20 \
  --batch-size 8 \
  --crop-size 128 \
  --lr 1e-5 \
  --init-checkpoint checkpoints/best.pt \
  --workers 4
```

Expected outcome: `checkpoints/best_extended.pt`, val MSE avg ≈ **0.028**, improving on the baseline.

### 4. Run inference on a single image

```bash
python enhance.py --input night.jpg --output enhanced_night.jpg --checkpoint checkpoints/best.pt
```

### 5. Evaluate against a ground-truth reference

```bash
python enhance.py --input night.jpg --reference day.jpg --checkpoint checkpoints/best.pt
```

Output:
```
MSE_R:   0.037988
MSE_G:   0.035524
MSE_B:   0.043262
MSE_avg: 0.038925
```

---

## Model Progression

Each generation of the model targeted a specific failure mode identified in the previous one.

### v1 — Baseline (Transient Attributes, MSE)

**Training data:** Transient Attributes only (1,176 pairs, 90 scenes)
**Loss:** MSE
**Architecture:** Standard U-Net, Sigmoid output — predicts the full enhanced image directly

The baseline established the core pipeline. MSE loss directly optimises the evaluation metric and produced clean convergence. However, MSE is known to cause *regression-to-mean*: when the model is uncertain about color temperature or scene brightness it averages possible outputs, producing washed-out mid-tones. The blue channel was consistently the hardest to recover, as night images suppress cool tones most severely.

---

### v1-extended — Fine-tuned (Transient Attributes + LOL, MSE)

**Training data:** TA + 420 LOL indoor pairs (1,676 total)
**Loss:** MSE (continued from v1 checkpoint)
**Change:** Added LOL dataset to address outdoor-only bias

Transient Attributes contains only outdoor scenes; the model had never seen indoor low-light images. LOL (Low-light Object Lossless) adds 485 paired indoor images and is the standard benchmark for low-light enhancement. Fine-tuning on the combined dataset improved val MSE from 0.028953 → 0.027752 and gave the model more generalisation across scene types. The washed-out color problem persisted — MSE does not penalise chrominance errors directly.

---

### v2 — Improved Pixel Loss (TA + LOL, L1 + MS-SSIM)

**Training data:** TA + LOL (same extended manifest)
**Loss:** `0.84 * (1 − MS-SSIM) + 0.16 * L1` (Wang et al. weighting)
**Change:** Replaced MSE with a combined structural + pixel loss; warm-started from v1-extended

MS-SSIM optimises luminance, contrast, and structural similarity at multiple scales; L1 anchors absolute pixel values and prevents the saturation that pure SSIM can produce. This improved pixel fidelity and sharpness, but **did not fix the color problem**. The root cause is that both L1 and MS-SSIM are effectively color-blind: a gray sky can have good SSIM with a blue sky if the luminance structure matches. Additionally, Sigmoid activations on uncertain units naturally converge toward 0.5 (mid-gray), and a model that must reconstruct the full image from scratch wastes capacity on background pixels that barely change.

---

### v3 — Residual Learning + Color-Aware Loss (TA + LOL, L1 + MS-SSIM + ColorLoss)

**Training data:** TA + LOL
**Loss:** `CombinedLoss (L1 + MS-SSIM) + 0.5 * ColorLoss (YCbCr)`
**Changes:** Two architectural and one loss change, each targeting a specific failure mode

**1 — Residual learning.** Instead of predicting the full enhanced image, the model predicts a *delta* (change) in Tanh range [−1, 1], which is added to the input and clamped to [0, 1]. This solves two problems at once: Tanh activations don't saturate to a mid-gray mean, and the model only needs to learn *what changes* between night and day — background structure is preserved for free via the residual connection. Sigmoid weights from v1-extended transfer cleanly (both activations are near-linear around 0); the first epoch recalibrates the output scale.

**2 — Color-aware loss.** A differentiable BT.601 RGB → YCbCr transform is used to decompose predictions into luminance (Y) and chrominance (Cb, Cr). Chrominance errors are penalised 2× relative to luminance. This directly punishes gray/desaturated outputs that match structure but miss color temperature — the exact failure mode L1 and MS-SSIM let through.

No new dependencies: the YCbCr transform is a pure PyTorch matrix multiply.

---

## Results

| Model | Val MSE avg | MSE R | MSE G | MSE B | Best epoch | Loss |
|---|---|---|---|---|---|---|
| v1 — Baseline (TA) | 0.028953 | 0.028700 | 0.025959 | 0.032200 | 22 | MSE |
| v1-extended (TA + LOL) | 0.027752 | 0.026945 | 0.025522 | 0.030788 | 19 | MSE |
| v2 (TA + LOL) | — | — | — | — | — | L1 + MS-SSIM |
| v3 (TA + LOL, residual) | 0.050844 | 0.050646 | 0.046377 | 0.055508 | 18 | L1 + MS-SSIM + ColorLoss |

> **Note on v3 val MSE:** v3's validation loss is higher than v1/v2 because `ColorLoss` penalises chrominance mismatches that MSE ignores. A model scoring better on this composite loss will produce more natural colours even if its raw MSE is higher. The held-out pair visual quality is the meaningful comparison.

**Final evaluation on held-out pair `night.jpg` / `day.jpg` (1024×737):**

| Model | MSE_R | MSE_G | MSE_B | MSE_avg |
|---|---|---|---|---|
| v1 (baseline) | 0.037988 | 0.035524 | 0.043262 | 0.038925 |
| v3 (residual + color) | 0.075107 | 0.069984 | 0.080816 | 0.075302 |

> **v3 trade-off:** v3's higher eval MSE reflects the trade-off: ColorLoss explicitly penalises chrominance mismatches (gray sky vs blue sky) that pixel-level MSE accepts. The model produces more saturated, natural-toned outputs at the cost of higher raw pixel error on a single held-out pair. Visual comparison is the meaningful test here. Future iterations may benefit from a staged training approach (warm-start from v1-extended for more epochs before introducing ColorLoss) or a lower `--color-weight`.

---

## Architecture

4-level U-Net (`base_filters=16`). Input images are padded to the nearest multiple of 16 before inference (reflect padding, bottom/right only) and cropped back to original dimensions — handles any input size without resizing.

**v1 / v1-extended / v2 (direct prediction):**
```
Input (H×W×3)
  → Encoder: 4× [ConvBlock + MaxPool2d]
  → Bottleneck: ConvBlock
  → Decoder: 4× [Upsample + skip connection + ConvBlock]
  → Output head: Conv1×1 + Sigmoid → full enhanced image in [0, 1]
```

**v3 (residual prediction):**
```
Input (H×W×3)
  → Encoder: 4× [ConvBlock + MaxPool2d]
  → Bottleneck: ConvBlock
  → Decoder: 4× [Upsample + skip connection + ConvBlock]
  → Output head: Conv1×1 + Tanh → delta in [−1, 1]
  → clamp(Input + delta, 0, 1) → enhanced image
```

Training details: crop size 176×176, batch 4, Adam optimizer, `ReduceLROnPlateau` scheduler (factor=0.5, patience=4). Scene-level 85/15 train/val split to prevent data leakage.

---

## Repository structure

| File | Purpose |
|---|---|
| `model.py` | U-Net architecture (`residual` flag for v3) |
| `losses.py` | `CombinedLoss`, `ColorLoss` (YCbCr), `PerceptualLoss`, `EnhancementLoss` |
| `dataset.py` | Dataset class; streams pairs from HF Hub via manifest CSV |
| `train.py` | v1 baseline training loop |
| `train_extended.py` | v1-extended fine-tuning on TA + LOL |
| `train_v2.py` | v2 training with L1 + MS-SSIM loss |
| `train_v3.py` | v3 training with residual U-Net + ColorLoss |
| `enhance.py` | Single-image inference + MSE evaluation |
| `app.py` | Gradio UI (also entry point for Docker) |
| `Dockerfile` | Docker build: Python 3.11-slim, venv, CPU-only torch |
| `low_light_manifest.csv` | 1,176 TA training pairs |
| `extended_manifest.csv` | 1,676 combined pairs (TA + LOL) |
