# Plan: v4 Improvement Roadmap

## Context

v3 suffers from two visual problems:
1. Output is too dark overall
2. Artificial lights (streetlamps, windows) are amplified instead of suppressed

Problem 2 is a Retinex problem: the model cannot distinguish a pixel bright from a streetlamp
vs. ambient daylight. All current losses (L1, MS-SSIM, ColorLoss) weight every pixel equally,
so the model is strongly rewarded for keeping bright lamp pixels bright. BrightnessLoss would
make this worse. CLAHE would make this worse.

---

## Solutions — Artificial Light Amplification

### 1. Input-Luminance-Weighted L1 — 30 min
`weight = 1 - clamp(Y_night, 0, 1)`, `loss = mean(weight * |pred - target|)`
Bright pixels (lamps) contribute almost no gradient. Dark ambient pixels dominate.
Add `WeightedL1Loss` to `losses.py`; `--input-weight` arg in `train_v4.py`.

### 2. Log-Domain L1 — 15 min
`L1(log(pred + ε), log(target + ε))`. Distance between 0.9 and 0.95 is tiny in log space;
distance between 0.05 and 0.1 is large. Auto down-weights high intensity pixels.
Replace L1 term inside `CombinedLoss` in `losses.py`.

### 3. Retinex Decomposition — 1–2 days (v5 path)
Two-branch network: `DecompNet → (R, L)`, `EnhanceNet(L) → L_enhanced`, output = R × L_enhanced.
Illumination smoothness constraint forces sharp bright spots (lamps) into reflectance R, not L.
L_enhanced is globally bright and smooth = daytime pattern.
Reference: RetinexNet (Wei et al., 2018), KinD (Zhang et al., 2019). Both CPU-feasible.
New branches in `model.py`; `IlluminationSmoothnessLoss` + `ReconstructionLoss` in `losses.py`.

### 4. Global Context Injection — 3–4 hrs
Compute mean/std/p10 per channel from full (uncropped) input image. Project to 16-dim
embedding, add to U-Net bottleneck. Lets the model identify pixels that are anomalously
bright vs. uniformly dark scene.
Changes: `model.py` (GlobalContextEncoder), `dataset.py` (return full-image stats).

---

## Solutions — Global Darkness (Without Worsening Lamps)

### 5. Staged ColorLoss — 15 min
`--color-warmup-epochs 5`: ColorLoss weight = 0 for first 5 epochs, then linear ramp.
Fixes gradient conflict from Tanh residual reorganization. Only `train_v4.py`.

### 6. Color Jitter + Gamma Augmentation — 45 min
Color jitter on low-light input only (not target). `ll_t = ll_t ** Uniform(0.6, 1.4)`.
Prevents overfitting to TA sensor color profile. `dataset.py` only.

### 7. Cosine LR Schedule — 5 min
Replace `ReduceLROnPlateau` with `CosineAnnealingWarmRestarts(T_0=10)`. `train_v4.py` only.

---

## NOT Feasible on CPU (Documented)

| Approach | Why Important | Why Not Feasible | Future Path |
|---|---|---|---|
| Diffusion (DDPM) | Models full distribution; naturally suppresses artificial lights | 30–60 min/image on CPU | HF `diffusers`, use U-Net as backbone |
| Restormer / LLFormer | Global attention sees whole image at once; SOTA on LOL | 40–80s/image on CPU | Pretrained weights; 2–4 GPU hrs to fine-tune |
| EnlightenGAN | Discriminator penalizes "night-looking" outputs; no paired data needed | 10–30× more compute; needs fast iteration | Compatible with existing dataset |
| CLIP perceptual loss | Semantically distinguishes night from day | 500ms/image; 6–10× slower epochs | Use when GPU available |

---

## Recommended v4

| Setting | v3 | v4 |
|---|---|---|
| Loss | CombinedLoss + 0.5×ColorLoss | WeightedL1 + log-L1 + MS-SSIM + ColorLoss (staged epoch 5) |
| Architecture | UNet(residual=True) | + GlobalContextEncoder |
| Augmentation | crop + h-flip | + color jitter + gamma |
| LR | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| Warm-start | best_extended.pt | best.pt (cleaner slate) |
| Epochs | 20 | 30 |

## Priority Order

| # | Change | Files | Fixes | Effort |
|---|---|---|---|---|
| 1 | WeightedL1Loss | `losses.py`, `train_v4.py` | Lamp amplification | 30 min |
| 2 | Log-L1 | `losses.py` | Lamp amplification | 15 min |
| 3 | Staged ColorLoss | `train_v4.py` | Gradient conflict | 15 min |
| 4 | Color jitter + gamma | `dataset.py` | Generalization | 45 min |
| 5 | Cosine LR | `train_v4.py` | Convergence | 5 min |
| 6 | Global context | `model.py`, `dataset.py` | Global brightness | 3–4 hrs |
| 7 | Retinex (v5) | `model.py`, `losses.py` | Lamp amplification (principled) | 1–2 days |

## Verification
1. Visual: streetlamp halos reduced vs v3 (`enhanced_night_v3.jpg`)
2. MSE_avg on `night.jpg / day.jpg` — should improve from v3's 0.075302
3. Compare lamp pixel value directly: `output[lamp_y, lamp_x]` lower than v3
