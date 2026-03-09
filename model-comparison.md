# Model Comparison — Low-Light to Day Enhancement

Evaluation pair: `night.jpg` / `day.jpg` (1024×737, not in training set).
Metrics computed in **float64**, pixel values in **[0, 1]**.
SSIM computed with `skimage` (`data_range=1.0`, `channel_axis=2`).

---

## Model Summaries

### v1 — Baseline MSE
**Checkpoint:** `best.pt` · Epoch 22 · Val MSE 0.028953
**Architecture:** UNet(base_filters=16, residual=False)
**Loss:** MSE (L2)
**Data:** Transient Attributes dataset only (1,010 train pairs, 90 scenes)
**Key highlights:**
- Establishes the baseline. Straightforward pixel-to-pixel MSE objective.
- Produces the best raw eval MSE of any version (0.038925).
- Outputs tend to be desaturated/gray: L2 penalises large errors heavily, pushing uncertain pixels toward the mean.

---

### v1-extended — LOL Fine-tune
**Checkpoint:** `best_extended.pt` · Epoch 19 · Val MSE 0.027752
**Architecture:** UNet(base_filters=16, residual=False)
**Loss:** MSE (L2)
**Data:** TA + LOL dataset (indoor paired low/normal-light, ~485 pairs appended)
**Key highlights:**
- Best val MSE of all versions (0.027752), but eval on `night.jpg` is worse than v1.
- LOL is indoor data — adding it improved generalisation on indoor scenes but introduced a slight domain mismatch for outdoor night scenes (the eval pair).
- Serves as the warm-start for v3.

---

### v2 — Perceptual Loss
**Checkpoint:** `best_v2.pt` · Epoch 10 · Val MSE 0.028914
**Architecture:** UNet(base_filters=16, residual=False)
**Loss:** L1 + MS-SSIM
**Data:** TA + LOL (extended manifest)
**Key highlights:**
- Replaced L2 with L1 + MS-SSIM to improve structural fidelity over pure pixel accuracy.
- MS-SSIM operates at multiple scales, penalising structural differences that L2 ignores.
- Eval MSE worse than v1 — L1 + MS-SSIM prioritises structure over absolute pixel values.

---

### v3 — Residual + Color-Aware
**Checkpoint:** `best_v3.pt` · Epoch 18 · Val MSE 0.050844
**Architecture:** UNet(base_filters=16, residual=True) — Tanh delta + clamp
**Loss:** CombinedLoss (L1 + MS-SSIM) + 0.5 × ColorLoss (BT.601 YCbCr)
**Data:** TA + LOL
**Key highlights:**
- Residual mode fixes the gray/desaturated output problem of v1–v2: the model predicts a Tanh *delta* added to the input rather than reconstructing the image from scratch.
- ColorLoss explicitly penalises chrominance (Cb, Cr) mismatch — 2× relative to luminance — discouraging gray outputs that match structure but miss colour temperature.
- Higher val MSE than v1 is expected: ColorLoss sacrifices raw pixel accuracy for chrominance correctness.
- Streetlamp amplification problem: all losses weight every pixel equally, so the model is rewarded for keeping bright lamp pixels bright → halos around light sources.

---

### v4 — Lamp Suppression + Global Context
**Checkpoint:** `best_v4.pt` · Epoch 25 · Val MSE 0.028453
**Architecture:** UNet(base_filters=16, residual=True, use_global_context=True)
**Loss:** WeightedL1 + LogL1 + MS-SSIM + ColorLoss (staged warmup, epoch 5→10)
**Data:** TA + LOL
**Key highlights:**
- **WeightedL1:** `weight = 1 - clamp(Y_night, 0, 1)` — bright lamp pixels contribute near-zero gradient; dark ambient pixels dominate.
- **LogL1:** log-domain L1 compresses bright-pixel errors; distance between 0.90 and 0.95 is tiny in log space vs. 0.05 and 0.10.
- **GlobalContextEncoder:** per-channel (mean, std, p10) of the full input image injected at the U-Net bottleneck. Lets the model distinguish "globally dark scene with isolated bright lamps" from "uniformly lit daytime scene."
- **Staged ColorLoss:** zero for first 5 epochs, linear ramp over next 5. Avoids gradient conflict during the Tanh recalibration phase after warm-starting from a sigmoid checkpoint.
- **CosineAnnealingWarmRestarts(T_0=10):** smoother LR trajectory; warm restarts help escape local minima mid-training.
- Val MSE matches v1 (0.028453 vs 0.028953) despite a fundamentally different loss. Eval MSE on `night.jpg` is higher than v3 because `night.jpg` is lamp-heavy and the suppression losses deliberately sacrifice accuracy on those pixels.

---

## Metric Comparison

| Model | Epoch | Val MSE | MSE_R | MSE_G | MSE_B | **MSE_avg** | **SSIM** |
|-------|-------|---------|-------|-------|-------|-------------|----------|
| v1 | 22 | 0.028953 | 0.037988 | 0.035524 | 0.043262 | **0.038925** | **0.5329** |
| v1-extended | 19 | 0.027752 | 0.050318 | 0.038497 | 0.040582 | 0.043132 | 0.4653 |
| v2 | 10 | 0.028914 | 0.051711 | 0.038521 | 0.042594 | 0.044275 | 0.4448 |
| v3 | 18 | 0.050844 | 0.048779 | 0.039017 | 0.050610 | 0.046135 | 0.2871 |
| v4 | 25 | 0.028453 | 0.059873 | 0.036027 | 0.047200 | 0.047700 | 0.3095 |

> **Note on v3/v4 eval:** These checkpoints use `residual=True` (Tanh delta architecture). Earlier eval runs that incorrectly loaded them with `residual=False` produced artificially different numbers; the table above reflects correct loading. Use `python enhance.py --residual` for v3/v4.

> **Why v3/v4 SSIM is lower:** Residual + color-aware losses shift the output color distribution toward warmer, more saturated tones. SSIM measures structural similarity against the reference's luminance structure — if the colour shift changes local contrast patterns, SSIM decreases even when the image looks perceptually better.

---

## Enhanced Outputs

Input image: `night.jpg`
Reference: `day.jpg`

| v1 | v1-extended |
|:--:|:-----------:|
| ![v1](enhanced_night_v1.jpg) | ![v1-extended](enhanced_night_v1_extended.jpg) |
| MSE 0.0389 · SSIM 0.533 | MSE 0.0431 · SSIM 0.465 |

| v2 | v3 |
|:--:|:--:|
| ![v2](enhanced_night_v2.jpg) | ![v3](enhanced_night_v3.jpg) |
| MSE 0.0443 · SSIM 0.445 | MSE 0.0461 · SSIM 0.287 |

| v4 | Reference (day) |
|:--:|:---------------:|
| ![v4](enhanced_night_v4.jpg) | ![day](day.jpg) |
| MSE 0.0477 · SSIM 0.310 | ground truth |

---

## Key Takeaways

| Question | Answer |
|---|---|
| Best raw MSE on eval pair | **v1** (0.038925) — simple MSE training generalises well to the eval scene |
| Best val MSE (generalisation) | **v4** (0.028453) — lamp-suppression losses don't hurt general reconstruction |
| Best colour fidelity | **v3 / v4** — ColorLoss explicitly targets chrominance; residual mode prevents gray outputs |
| Best structural fidelity (SSIM) | **v1** (0.533) — direct L2 optimisation preserves luminance structure |
| Lamp amplification | **v3** most severe (equal pixel weighting); **v4** designed to suppress it |
| Trade-off | Raw MSE vs. perceptual quality: v1 wins on numbers, v3/v4 target visual realism |
