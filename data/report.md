# Project Report — Night-to-Day Image Enhancement

**Author:** Taisiia Yakovenko
**Date:** March 2026
**GitHub:** [tyakovenko/night-to-day-imgConversion](https://github.com/tyakovenko/night-to-day-imgConversion)
**HuggingFace:** [tyakovenko](https://huggingface.co/tyakovenko)

---

## 1. Problem Statement

Enhance a night-time image to match the appearance of the same scene captured during the day. Evaluation metric: **channel-wise MSE** (R, G, B then averaged), computed in float64 with pixel values in [0, 1]. The day image is used only for training supervision — never copied or blended at inference.

---

## 2. Dataset

**Primary: Transient Attributes Dataset** (Laffont et al., Brown University)
8,571 images across 101 outdoor scenes, each annotated with 40 continuous attributes.

Pair selection:
- **Day target:** `daylight > 0.8`, highest-scoring image per scene
- **Low-light input:** `night > 0.5` OR (`dark > 0.5` AND `daylight < 0.4`) OR (`dawndusk > 0.5` AND `daylight < 0.4`)

Result: **1,176 validated pairs across 90 scenes**, stored in `low_light_manifest.csv`.
Train/val split is scene-level (85/15) to prevent data leakage: 1,010 train pairs / 166 val pairs.

**Supplemental: LOL Dataset** (~485 indoor paired low/normal-light images)
Added for v1-extended, v2, v3, v4. Combined manifest: `extended_manifest.csv` (1,676 pairs).

All data hosted on HuggingFace Hub: [`tyakovenko/night-to-day-enhancement`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) (2,537 files, ~256 MB).

---

## 3. Architecture

All versions share the same base **4-level U-Net** encoder-decoder, implemented in PyTorch.

```
Input (3, H, W)
  └─ Encoder:    [16] → [32] → [64] → [128]  (Conv-BN-ReLU × 2 + MaxPool)
  └─ Bottleneck: [256]
  └─ Decoder:    [128] → [64] → [32] → [16]  (Upsample + skip concat + Conv-BN-ReLU × 2)
  └─ Output:     Conv 1×1 → Sigmoid (or Tanh delta in residual mode)
```

| Property | Value |
|---|---|
| Parameters | 1,811,811 |
| Upsampling | Bilinear + 1×1 conv (no checkerboard artifacts) |
| `residual=True` | Model predicts a Tanh delta added to the input instead of reconstructing from scratch |
| `use_global_context=True` (v4) | Injects per-channel (mean, std, p10) of the full input at the bottleneck |

Inference handles non-multiple-of-16 image sizes via reflect padding (bottom/right only), cropped back before metric computation.

---

## 4. Models

Five model versions were trained, each building on the last.

### v1 — Baseline MSE
**Checkpoint:** `best.pt` · Epoch 22 · `tyakovenko/night-to-day-enhancement-model`
**Loss:** MSE (L2) · **Data:** Transient Attributes only

| Strength | Weakness |
|---|---|
| Best eval MSE on the provided pair (0.0389) | Outputs tend to be desaturated/gray — L2 averages uncertain pixels toward the mean |
| Best SSIM (0.533) — direct L2 preserves luminance structure | Struggles with color temperature recovery |
| Simplest, most stable training | No mechanism to distinguish lamp pixels from ambient light |

---

### v1-extended — LOL Fine-Tune
**Checkpoint:** `best_extended.pt` · Epoch 19 · same repo as v1
**Loss:** MSE (L2) · **Data:** TA + LOL · **Init:** `best.pt`

| Strength | Weakness |
|---|---|
| Best *val* MSE overall (0.027752) | Eval MSE *worse* than v1 on the outdoor eval pair (0.0431 vs 0.0389) |
| Improved generalisation on indoor scenes (warm artificial lighting) | LOL is indoor — slight domain mismatch for outdoor night scenes |

---

### v2 — Structural Loss
**Checkpoint:** `best_v2.pt` · Epoch 10 · `tyakovenko/night-to-day-enhancement-model-v2`
**Loss:** L1 + MS-SSIM · **Data:** TA + LOL · **Init:** `best_extended.pt`

| Strength | Weakness |
|---|---|
| MS-SSIM penalises structural differences L2 ignores | Eval MSE worse than v1 (0.0443) — structural loss doesn't directly optimise the MSE metric |
| L1 more robust to outlier pixels than L2 | No improvement in color fidelity over v1 |

---

### v3 — Residual + Color-Aware
**Checkpoint:** `best_v3.pt` · Epoch 18 · `tyakovenko/night-to-day-enhancement-model-v3`
**Loss:** L1 + MS-SSIM + ColorLoss (BT.601 YCbCr, 2× chrominance weight) · **Data:** TA + LOL · **Init:** `best_extended.pt`

| Strength | Weakness |
|---|---|
| Residual mode eliminates gray/desaturated outputs | Highest val MSE (0.0508) — ColorLoss trades pixel accuracy for chrominance correctness |
| ColorLoss explicitly penalises Cb/Cr mismatch | Equal pixel weighting amplifies streetlamp artifacts (halos around light sources) |
| Warmer, more visually saturated outputs | Lowest SSIM (0.287) — color shift alters local contrast patterns |

> **Usage note:** `enhance.py --residual` required for correct v3/v4 inference.

---

### v4 — Lamp Suppression + Global Context
**Checkpoint:** `best_v4.pt` · Epoch 25 · `tyakovenko/night-to-day-enhancement-model-v4`
**Loss:** WeightedL1 + LogL1 + MS-SSIM + ColorLoss (staged warmup epochs 5→10)
**Data:** TA + LOL · **Init:** `best.pt`

| Strength | Weakness |
|---|---|
| WeightedL1 downweights bright lamp pixels (`weight = 1 − clamp(Y_night)`) — fixes halo artifacts from v3 | Eval MSE on the lamp-heavy eval pair is highest (0.0477) — by design; lamp pixels are deliberately deprioritised |
| LogL1 compresses bright-pixel errors in log space | Higher complexity: 3 loss components + staged warmup + cosine LR |
| GlobalContextEncoder distinguishes "dark scene + bright lamps" from "daytime" at the bottleneck | Global context adds inference overhead (per-channel stats computed over full image) |
| Staged ColorLoss avoids gradient conflict during Tanh recalibration after warm-starting from a sigmoid checkpoint | |
| Val MSE matches v1 (0.028453) despite a fundamentally different loss objective | |

---

## 5. Results Summary

Eval pair: `night.jpg` / `day.jpg` (1024×737, out-of-distribution — not in training set).

| Model | Val MSE | MSE_R | MSE_G | MSE_B | **MSE_avg** | **SSIM** |
|---|---|---|---|---|---|---|
| v1 | 0.028953 | 0.037988 | 0.035524 | 0.043262 | **0.038925** | **0.5329** |
| v1-extended | 0.027752 | 0.050318 | 0.038497 | 0.040582 | 0.043132 | 0.4653 |
| v2 | 0.028914 | 0.051711 | 0.038521 | 0.042594 | 0.044275 | 0.4448 |
| v3 | 0.050844 | 0.048779 | 0.039017 | 0.050610 | 0.046135 | 0.2871 |
| v4 | 0.028453 | 0.059873 | 0.036027 | 0.047200 | 0.047700 | 0.3095 |

**Which model for which use case:**

| Goal | Recommended model |
|---|---|
| Lowest raw MSE / best pixel accuracy | **v1** |
| Best generalisation across diverse scenes (val MSE) | **v4** |
| Most visually saturated / realistic color | **v3 or v4** |
| Indoor scenes (artificial lighting) | **v1-extended** |
| Scenes with streetlamps / isolated light sources | **v4** (lamp suppression) |

**Channel-wise pattern:** Blue channel MSE is consistently highest — night images suppress cool tones most severely. Green channel MSE is lowest throughout.

---

## 6. Inference

```bash
# Basic enhancement
python enhance.py --input night.jpg --output enhanced.jpg

# With MSE evaluation against ground truth
python enhance.py --input night.jpg --reference day.jpg --output enhanced.jpg

# v3 or v4 checkpoint (residual mode required)
python enhance.py --input night.jpg --checkpoint checkpoints/best_v3.pt --residual
python enhance.py --input night.jpg --checkpoint checkpoints/best_v4.pt --residual
```

The pipeline handles arbitrary image sizes via reflect padding to the nearest multiple of 16 (required by 4× MaxPool). Padded pixels are cropped before saving and before MSE computation.

---

## 7. HuggingFace Deployment

| Resource | Link |
|---|---|
| Dataset | [`tyakovenko/night-to-day-enhancement`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) |
| Model v1 / v1-ext | [`tyakovenko/night-to-day-enhancement-model`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model) |
| Model v2 | [`tyakovenko/night-to-day-enhancement-model-v2`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v2) |
| Model v3 | [`tyakovenko/night-to-day-enhancement-model-v3`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v3) |
| Model v4 | [`tyakovenko/night-to-day-enhancement-model-v4`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v4) |
| Live Demo | [`tyakovenko/night-to-day-enhancement` (Space)](https://huggingface.co/spaces/tyakovenko/night-to-day-enhancement) |

All checkpoints are publicly accessible (no authentication required). The Gradio Space supports all five models via a dropdown and accepts an optional reference image for MSE reporting.

---

## 8. Future Work

### Short-term (incremental improvements)

| Idea | Expected impact |
|---|---|
| **Larger base_filters (32 or 64)** | More model capacity for detail recovery; currently limited to 16 by CPU training budget. Requires GPU. |
| **Larger crop sizes (256×256 or 512×512)** | Training on 128×128 crops means the model never sees full-image context during training. Larger crops improve global color consistency. |
| **LPIPS perceptual loss** | Replace or supplement MSE with Learned Perceptual Image Patch Similarity — better alignment with human visual quality judgement than pixel-level metrics. |
| **Stricter early stopping** | Val MSE oscillates across all versions; a patience-based stop with a held-out shadow set would prevent saving a noise spike as the best checkpoint. |

### Medium-term (architectural changes)

| Idea | Expected impact |
|---|---|
| **Attention gates in skip connections** | Suppress irrelevant encoder features from being passed to the decoder — addresses the lamp amplification problem at the architecture level rather than through loss weighting. |
| **GAN discriminator (pix2pix-style)** | A patch discriminator would push outputs toward photorealistic textures rather than MSE-blurred averages. Trade-off: harder to train, may increase MSE even as perceptual quality improves. |
| **Denoising as a separate stage** | Night images contain significant noise; preprocessing with a dedicated denoiser (e.g., DnCNN) before enhancement could improve both MSE and visual quality. |

### Long-term (data and generalisation)

| Idea | Expected impact |
|---|---|
| **More diverse outdoor data** | The TA dataset covers 90 scenes. Scaling to 500+ scenes with greater geographic/lighting diversity would improve generalisation to unseen scenes. |
| **Video-based training** | Temporal consistency matters for real use cases. Training on video pairs with optical flow alignment would reduce flicker in sequential frames. |
| **Domain-specific fine-tuning** | Fine-tune a base model on target-domain images (e.g., drone footage, dashcam, surveillance) for deployment-specific performance gains. |

---

## 9. Reproducibility

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# Train
python v1/train.py
python v1/train_extended.py
python v2/train_v2.py
python v3/train_v3.py
python v4/train_v4.py

# Inference + eval
python enhance.py --input night.jpg --reference day.jpg
```

Load any checkpoint from HuggingFace:

```python
from huggingface_hub import hf_hub_download
import torch
from model import UNet

ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model", "best.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = UNet(base_filters=ckpt["args"]["base_filters"])
model.load_state_dict(ckpt["model"])
model.eval()
```

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*. https://arxiv.org/abs/1505.04597
2. Laffont, P.-Y., Ren, Z., Tao, X., Qian, C., & Hays, J. (2014). Transient Attributes for High-Level Understanding and Editing of Outdoor Scenes. *ACM SIGGRAPH 2014*. http://transattr.cs.brown.edu/
3. Wei, C., Wang, W., Yang, W., & Liu, J. (2018). Deep Retinex Decomposition for Low-Light Enhancement. *BMVC 2018*. (LOL Dataset) https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
