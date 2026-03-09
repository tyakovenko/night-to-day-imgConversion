---
title: Night to Day Enhancement
emoji: 🌙
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
- **v4 model:** [tyakovenko/night-to-day-enhancement-model-v4](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v4) — best_v4.pt (WeightedL1 + LogL1 + GlobalContext)

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

### 2. Train v4 (recommended — lamp suppression + global context)

```bash
python train_v4.py \
  --epochs 30 \
  --batch-size 4 \
  --crop-size 176 \
  --lr 1e-4 \
  --color-warmup-epochs 5 \
  --global-context \
  --augment-color \
  --init-checkpoint checkpoints/best.pt \
  --workers 4
```

Expected outcome: `checkpoints/best_v4.pt` at ~epoch 25, val MSE avg ≈ **0.028**.
Runtime: ~220s/epoch on CPU.

### 3. Train the baseline model (v1 — Transient Attributes, MSE)

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

### 4. Run inference on a single image

```bash
# v1 / v1-extended / v2 (non-residual):
python enhance.py --input night.jpg --checkpoint checkpoints/best.pt

# v3 / v4 (residual=True — must pass --residual flag):
python enhance.py --input night.jpg --checkpoint checkpoints/best_v3.pt --residual
python enhance.py --input night.jpg --checkpoint checkpoints/best_v4.pt --residual
```

### 5. Evaluate against a ground-truth reference

```bash
python enhance.py --input night.jpg --reference day.jpg \
  --checkpoint checkpoints/best_v4.pt --residual
```

---

## Model Progression

Each generation targeted a specific failure mode identified in the previous one.

### v1 — Baseline (Transient Attributes, MSE)

**Loss:** MSE · **Architecture:** UNet(residual=False)

The baseline established the core pipeline. MSE directly optimises the evaluation metric and produces clean convergence. Produces the best raw eval MSE of any version. Outputs tend to be desaturated/gray: MSE pushes uncertain pixels toward the mean. The blue channel was consistently the hardest to recover.

---

### v1-extended — LOL Fine-tune (TA + LOL, MSE)

**Loss:** MSE · **Change:** Added 420 indoor LOL pairs

Improved generalisation across scene types; best val MSE overall (0.027752). LOL is indoor data — slight domain mismatch for the outdoor eval pair means eval MSE is actually worse than v1 on `night.jpg`. The washed-out color problem persisted.

---

### v2 — Perceptual Loss (TA + LOL, L1 + MS-SSIM)

**Loss:** `0.84 × (1 − MS-SSIM) + 0.16 × L1` (Wang et al.) · **Architecture:** UNet(residual=False)

MS-SSIM optimises structural similarity at multiple scales; L1 anchors absolute pixel values. Both losses are effectively color-blind — a gray sky matches a blue sky if luminance structure agrees. Sigmoid activations on uncertain units still converge toward mid-gray.

---

### v3 — Residual + Color-Aware (TA + LOL, L1 + MS-SSIM + ColorLoss)

**Loss:** `CombinedLoss + 0.5 × ColorLoss` · **Architecture:** UNet(residual=True)

**Residual learning:** model predicts a Tanh delta [−1, 1] added to the input and clamped. Fixes gray outputs — the model only learns *what changes*, background is preserved via the skip. **ColorLoss:** differentiable BT.601 RGB → YCbCr; chrominance (Cb, Cr) penalised 2× vs luminance. Directly punishes desaturated outputs. Remaining problem: all losses still weight every pixel equally, so bright lamp pixels dominate gradients → streetlamp halos amplified.

---

### v4 — Lamp Suppression + Global Context (TA + LOL, WeightedL1 + LogL1 + MS-SSIM + ColorLoss)

**Loss:** `WeightedL1 + LogL1 + MS-SSIM + ColorLoss` · **Architecture:** UNet(residual=True, use_global_context=True)

**WeightedL1:** `weight = 1 − clamp(Y_night, 0, 1)` — bright lamp pixels contribute near-zero gradient; dark ambient pixels dominate. **LogL1:** log-domain L1 compresses bright-pixel errors; distance between 0.90 and 0.95 is tiny in log space vs. 0.05 and 0.10. **GlobalContextEncoder:** per-channel (mean, std, p10) of the full input image injected at the bottleneck — lets the model distinguish "globally dark scene with isolated bright lamp" from "uniformly lit daytime scene." **Staged ColorLoss:** zero for epochs 1–5, linear ramp over epochs 6–10, avoids gradient conflict during warm-start recalibration. **LR:** `CosineAnnealingWarmRestarts(T_0=10)`.

---

## Results

### Validation MSE (on held-out scenes from training data)

| Model | Val MSE avg | Best epoch | Loss |
|-------|-------------|------------|------|
| v1 — Baseline (TA) | 0.028953 | 22 | MSE |
| v1-extended (TA + LOL) | **0.027752** | 19 | MSE |
| v2 (TA + LOL) | 0.028914 | 10 | L1 + MS-SSIM |
| v3 (TA + LOL, residual) | 0.050844 | 18 | L1 + MS-SSIM + ColorLoss |
| v4 (TA + LOL, residual + global ctx) | 0.028453 | 25 | WeightedL1 + LogL1 + MS-SSIM + ColorLoss |

### Evaluation on held-out pair `night.jpg` / `day.jpg` (1024×737)

| Model | MSE_R | MSE_G | MSE_B | MSE_avg | SSIM |
|-------|-------|-------|-------|---------|------|
| v1 — Baseline | 0.037988 | 0.035524 | 0.043262 | **0.038925** | **0.5329** |
| v1-extended | 0.050318 | 0.038497 | 0.040582 | 0.043132 | 0.4653 |
| v2 | 0.051711 | 0.038521 | 0.042594 | 0.044275 | 0.4448 |
| v3 (residual + color) | 0.048779 | 0.039017 | 0.050610 | 0.046135 | 0.2871 |
| v4 (lamp suppression) | 0.059873 | 0.036027 | 0.047200 | 0.047700 | 0.3095 |

> **Trade-off:** v1 wins on raw MSE and SSIM. v3/v4 sacrifice pixel accuracy on bright lamp pixels (which dominate `night.jpg`) for better colour fidelity and lamp suppression. The right choice depends on whether the evaluation is metric-based or perceptual.

> **Note:** v3/v4 use `residual=True`. Pass `--residual` to `enhance.py` for correct inference — these checkpoints predate that flag being saved in args.

---

## Architecture

4-level U-Net (`base_filters=16`). Input images are padded to the nearest multiple of 16 before inference (reflect padding, bottom/right only) and cropped back — handles any input size.

**v1 / v1-extended / v2 (direct prediction):**
```
Input → Encoder (4× ConvBlock + MaxPool) → Bottleneck → Decoder (4× Upsample + skip) → Conv1×1 + Sigmoid
```

**v3 (residual prediction):**
```
Input → Encoder → Bottleneck → Decoder → Conv1×1 + Tanh → delta
Output = clamp(Input + delta, 0, 1)
```

**v4 (residual + global context):**
```
Input → Encoder → Bottleneck ──┐
                               ├─ add → Decoder → Conv1×1 + Tanh → delta
GlobalContextEncoder(stats) ───┘
Output = clamp(Input + delta, 0, 1)

GlobalContextEncoder: [mean_R, mean_G, mean_B, std_R, std_G, std_B, p10_R, p10_G, p10_B]
                      → Linear(9→32) → ReLU → Linear(32→bottleneck_channels) → broadcast
```

---

## Repository Structure

| File | Purpose |
|------|---------|
| `model.py` | U-Net + `GlobalContextEncoder`; `residual` (v3+) and `use_global_context` (v4) flags |
| `losses.py` | `CombinedLoss`, `ColorLoss`, `WeightedL1Loss`, `LogL1Loss`, `V4Loss`, `PerceptualLoss` |
| `dataset.py` | Dataset class; streams pairs from HF Hub via manifest CSV; gamma augmentation; global stats |
| `train.py` | v1 baseline training loop |
| `train_extended.py` | v1-extended fine-tuning (TA + LOL) |
| `train_v2.py` | v2 — L1 + MS-SSIM loss |
| `train_v3.py` | v3 — residual U-Net + ColorLoss |
| `train_v4.py` | v4 — lamp suppression losses + global context + cosine LR |
| `enhance.py` | Single-image inference + MSE evaluation (`--residual` flag for v3/v4) |
| `app.py` | Gradio UI (also Docker entry point); v4 default |
| `Dockerfile` | Python 3.11-slim, venv, CPU-only torch |
| `low_light_manifest.csv` | 1,176 TA training pairs |
| `extended_manifest.csv` | 1,676 combined pairs (TA + LOL) |
| `model-comparison.md` | Full metric comparison + enhanced output images for all versions |
| `report.md` | Full report summarizing the application |
