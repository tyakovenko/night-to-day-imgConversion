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
- **Extended dataset:** `extended_manifest.csv` in this repo — adds 420 LOL pairs (same HF dataset repo, extended locally)
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

### 2. Train the baseline model (v1 — Transient Attributes, MSE)

Training streams images from HF Hub on first run (~1,266 files, cached locally after).

```bash
python v1/train.py \
  --epochs 30 \
  --batch-size 8 \
  --crop-size 128 \
  --base-filters 16 \
  --lr 1e-4 \
  --workers 4
```

Expected outcome: `checkpoints/best.pt` at ~epoch 22, val MSE avg ≈ **0.029**.
Runtime: ~90–110s/epoch on CPU.

### 3. Train v4 (recommended — lamp suppression + global context)

Requires `checkpoints/best.pt` from step 2 as the warm-start checkpoint.

```bash
python v4/train_v4.py \
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

Each generation targeted a specific failure mode identified in the previous one. See [`data/report.md`](data/report.md) for full per-model analysis, strength/weakness breakdown, and results.

| Version | Key change | Best eval MSE | Best SSIM |
|---------|-----------|--------------|-----------|
| v1 | Baseline MSE on Transient Attributes | **0.0389** | **0.533** |
| v1-extended | Fine-tuned on TA + LOL indoor data | 0.0431 | 0.465 |
| v2 | Replaced MSE with L1 + MS-SSIM | 0.0443 | 0.445 |
| v3 | Residual U-Net + ColorLoss (fixed gray outputs) | 0.0461 | 0.287 |
| v4 | Lamp suppression losses + GlobalContextEncoder | 0.0477 | 0.310 |

> v3/v4 use `residual=True` — pass `--residual` to `enhance.py` for correct inference.

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
| `v1/train.py` | v1 baseline training loop |
| `v1/train_extended.py` | v1-extended fine-tuning (TA + LOL) |
| `v2/train_v2.py` | v2 — L1 + MS-SSIM loss |
| `v3/train_v3.py` | v3 — residual U-Net + ColorLoss |
| `v4/train_v4.py` | v4 — lamp suppression losses + global context + cosine LR |
| `enhance.py` | Single-image inference + MSE evaluation (`--residual` flag for v3/v4) |
| `app.py` | Gradio UI (also Docker entry point); v4 default |
| `Dockerfile` | Python 3.11-slim, venv, CPU-only torch |
| `low_light_manifest.csv` | 1,176 TA training pairs |
| `extended_manifest.csv` | 1,676 combined pairs (TA + LOL) |
| `model-comparison.md` | Full metric comparison + enhanced output images for all versions |
| `data/report.md` | Full report summarizing the application |

## Generative AI Use Disclaimer

Claude Code (Anthropic) was used as the primary AI assistant for code scaffolding, training loop implementation, and documentation. All architectural decisions, hyperparameter choices, and experimental results are the author's own. Dataset curation, threshold selection, and model evaluation were performed and verified independently. All final reports and code were reviewed.
