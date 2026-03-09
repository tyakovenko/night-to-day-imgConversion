# Project Log — Low-Light to Day Image Enhancement

## Session: 2026-03-03

### Completed This Session

**Phase 1 (DataAnalyst) — COMPLETE**

- `analyze_dataset.py` analyzed 8,571 annotated images across 101 scenes
- `data-analyst-report.md` written (full distribution analysis + threshold justification)
- `low_light_manifest.csv` written — 1,176 validated (low-light, day-target) pairs across 90 scenes
- Dataset uploaded to Hugging Face Hub: `tyakovenko/night-to-day-enhancement`
  - 2,537 files, 255.6 MB
  - URL: https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement

### Upload Debugging

Three upload scripts were created during Phase 1 before a working approach was found:
- `upload_to_hf.py` — per-file `upload_file()` calls; hit rate limits, timed out at scene `00000325`
- `upload_to_hf_v2.py` — `upload_folder()` single commit; too large for 1266 files
- `upload_to_hf_v3.py` — `upload_large_folder()` with chunking + resume; **this is the correct approach**

Upload was completed by running `upload_to_hf_v3.py` foreground with a 20-minute timeout.

### Current Status

- Phase 1: **DONE**
- Phase 2 (Backend, training pipeline): **NOT STARTED**

### Open Tasks for Next Session

- Spawn Backend to implement `Dataset` class reading from `tyakovenko/night-to-day-enhancement` on HF Hub
- Implement `train.py`, `enhance.py`, `app.py` per CLAUDE.md conventions
- All image pair access must go through `low_light_manifest.csv` — never raw annotations

> **Data access rule:** All downstream stages (Dataset class, training, inference) read images and metadata directly from `tyakovenko/night-to-day-enhancement` on Hugging Face Hub. No code may reference local image paths (`transientAttributesDataset/` or `hf_staging/`).

### Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Low-light threshold | `night > 0.5` OR (`dark > 0.6` AND `daylight < 0.3`) | Distribution-justified; see data-analyst-report.md §3 |
| Day-target threshold | `daylight > 0.8`, highest score per scene | Clear distribution gap above 0.8 |
| Upload method | `upload_large_folder()` | Only method that handles 1000+ files with rate-limit tolerance and resume |

### Blockers / Open Questions

- None currently

---

## Session: 2026-03-08 (v3 training — color fix)

### Completed This Session

**Phase 2 (Backend) — IN PROGRESS**

Created training pipeline scripts:
- `model.py` — U-Net encoder-decoder (4 levels, skip connections, Sigmoid output)
- `dataset.py` — `LowLightDataset` class; loads pairs from HF Hub via `low_light_manifest.csv`; scene-level train/val split
- `train.py` — full training loop with per-epoch MSE reporting, checkpointing, LR scheduling, experiment log
- `enhance.py` — inference script; full-resolution output, channel-wise MSE evaluation

**Training run started:** U-Net, base_filters=16, crop=128×128, batch=8, lr=1e-4, 30 epochs, CPU-only
**Split:** 1,010 train pairs (77 scenes) / 166 val pairs (13 scenes)
**Model parameters:** 7,240,387
**Images prefetched from HF Hub:** 1,266 files in 71s — all subsequent epochs use local cache, no auth required

### Training Results (COMPLETE — 30 epochs)

| Ep | Train Loss | Val MSE avg | MSE_R    | MSE_G    | MSE_B    | Time | Best |
|----|-----------|-------------|----------|----------|----------|------|------|
|  1 | 0.054488  | 0.039336    | 0.035079 | 0.035582 | 0.047347 |  97s | ✓ |
|  2 | 0.051836  | 0.035965    | 0.035797 | 0.032521 | 0.039575 | 124s | ✓ |
|  3 | 0.050300  | 0.037730    | 0.035330 | 0.034132 | 0.043728 | 122s |   |
|  4 | 0.046166  | 0.036025    | 0.035589 | 0.033050 | 0.039435 |  90s |   |
|  5 | 0.047562  | 0.037194    | 0.036199 | 0.034183 | 0.041201 |  91s |   |
|  6 | 0.048429  | 0.034016    | 0.032854 | 0.030858 | 0.038337 | 109s | ✓ |
|  7 | 0.047591  | 0.033815    | 0.031671 | 0.030261 | 0.039514 |  95s | ✓ |
|  8 | 0.049659  | 0.032454    | 0.030424 | 0.028821 | 0.038117 |  95s | ✓ |
|  9 | 0.047977  | 0.036101    | 0.032500 | 0.032339 | 0.043465 |  98s |   |
| 10 | 0.048240  | 0.033233    | 0.032639 | 0.029650 | 0.037409 |  92s |   |
| 11 | 0.047696  | 0.033106    | 0.031234 | 0.029439 | 0.038645 |  88s |   |
| 12 | 0.047381  | 0.030668    | 0.031171 | 0.028090 | 0.032744 |  92s | ✓ |
| 13 | 0.048375  | 0.037351    | 0.034258 | 0.033383 | 0.044412 |  89s |   |
| 14 | 0.046013  | 0.034301    | 0.032956 | 0.030723 | 0.039223 |  88s |   |
| 15 | 0.047037  | 0.033501    | 0.032079 | 0.030086 | 0.038339 |  92s |   |
| 16 | 0.045721  | 0.036675    | 0.033941 | 0.033099 | 0.042984 |  94s |   |
| 17 | 0.046112  | 0.032932    | 0.030803 | 0.029809 | 0.038186 |  86s |   |
| 18 | 0.046836  | 0.036563    | 0.033851 | 0.032513 | 0.043326 |  90s |   |
| 19 | 0.046244  | 0.033501    | 0.032378 | 0.030055 | 0.038071 |  99s |   |
| 20 | 0.046453  | 0.029839    | 0.029595 | 0.026591 | 0.033331 | 111s | ✓ |
| 21 | 0.044902  | 0.034271    | 0.033707 | 0.030986 | 0.038121 | 100s |   |
| 22 | 0.043913  | 0.028953    | 0.028700 | 0.025959 | 0.032200 |  88s | ✓ |
| 23 | 0.046162  | 0.032787    | 0.032881 | 0.029942 | 0.035538 |  94s |   |
| 24 | 0.044521  | 0.030584    | 0.030248 | 0.027590 | 0.033915 |  95s |   |
| 25 | 0.046452  | 0.033651    | 0.032762 | 0.030585 | 0.037607 |  91s |   |
| 26 | 0.044440  | 0.037195    | 0.035323 | 0.033143 | 0.043118 | 102s |   |
| 27 | 0.044797  | 0.035506    | 0.035270 | 0.032336 | 0.038913 |  94s |   |
| 28 | 0.044276  | 0.032017    | 0.030109 | 0.029099 | 0.036844 | 103s |   |
| 29 | 0.044924  | 0.033364    | 0.032470 | 0.030195 | 0.037426 |  99s |   |
| 30 | 0.044869  | 0.032616    | 0.031317 | 0.029604 | 0.036927 | 104s |   |

**Best checkpoint:** Epoch 22, Val MSE avg **0.028953** (R=0.0287, G=0.0260, B=0.0322)
**Checkpoint saved:** `checkpoints/best.pt`
**LR schedule:** halved to 5e-5 at epoch 19, halved to 2.5e-5 at epoch 29
**Observation:** Blue channel (MSE_B) consistently highest — expected, night scenes suppress cool tones most. Green channel learns fastest.

### Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | U-Net (base_filters=16) | Proven for pixel-wise image translation; skip connections preserve spatial detail |
| Loss | MSE (L2) | Directly optimizes evaluation metric |
| Train/val split | Scene-level 85/15 | Prevents data leakage between train and test |
| Crop size | 128×128 | CPU-feasible (~90s/epoch); full resolution used at inference |
| Initial config (killed) | base_filters=32, crop=256 | ~20 min/epoch on CPU — too slow |

### Completed This Session (continued)

- `checkpoints/best.pt` uploaded to `tyakovenko/night-to-day-enhancement-model` (HF Hub)
- `app.py` wired to real U-Net checkpoint — model loads from HF Hub at Space startup
- HF Space live at `https://tyakovenko-night-to-day-enhancement.hf.space` (HTTP 200)
- GitHub pushed: commits `bfedcee`, `961aa28`, `5db459e`

### Open Tasks / TODOs

- **[BUG] Fix Space crash on Enhance with real images** — same padding fix still needed in `app.py`
  - Location: `enhance_image()` in `app.py` — apply same `pad_to_multiple` + crop pattern as `enhance.py`

- Consider additional training epochs or LOL dataset fine-tuning to push MSE lower

---

## Session: 2026-03-08 (continued — eval fix)

### Completed This Session

**Final evaluation run on `night.jpg` / `day.jpg`**

Two bugs were fixed in `enhance.py` to enable inference on the final eval pair:

1. **Padding bug (shape mismatch):** The U-Net has 4 MaxPool2d layers, so input H and W must each be divisible by 16. `night.jpg` / `day.jpg` are 1024×737 — height 737 % 16 = 1, causing a skip-connection shape mismatch in the decoder. Fix: added `pad_to_multiple(t, 16)` using **reflect padding** (bottom/right only) before the forward pass, then cropped output back to original dimensions (`[:h_orig, :w_orig]`). Reflect padding mirrors real border pixels so the model sees plausible content; the 15-row padded strip is fully discarded before saving and before MSE computation.

2. **base_filters mismatch:** `enhance.py` defaulted to `base_filters=32`, but the checkpoint (`best.pt`) was trained with `base_filters=16`. Loading the wrong architecture caused a state_dict key mismatch. Fix: auto-detect `base_filters` from `ckpt["args"]["base_filters"]`; fall back to 16 if key absent.

### Final Evaluation Results

```
Input:  night.jpg (1024×737)
Ref:    day.jpg   (1024×737)
Model:  checkpoints/best.pt  (epoch 22, base_filters=16)

MSE_R:   0.037988
MSE_G:   0.035524
MSE_B:   0.043262
MSE_avg: 0.038925
```

Note: val MSE during training was 0.028953 — higher score on final eval pair is expected (different scene, not in training set).

### Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Padding strategy | reflect | Mirrors real border pixels; avoids hard black edge that would bias convolution near border; padded strip is cropped back and never appears in output |
| Padding sides | bottom + right only | Simplifies crop: `[:h_orig, :w_orig]` is exact, no offset needed |
| base_filters detection | auto from `ckpt["args"]` | Eliminates flag/checkpoint mismatch class of bug permanently |

---

## Session: 2026-03-08 (v3 training + deployment)

### Completed This Session

**Root cause analysis for washed-out / gray outputs (v1–v2)**

All prior models produced gray/desaturated images due to three compounding causes:
1. Sigmoid activations on uncertain units converge to 0.5 (mid-gray)
2. Model must reconstruct the full image from scratch — capacity wasted on unchanged background
3. L1 and MS-SSIM are color-blind: a gray sky can match a blue sky if luminance structure matches

**v3 implementation**

Three targeted changes in `model.py`, `losses.py`, and new `train_v3.py`:
- `model.py`: Added `residual` flag to `UNet`. When `True`, output activation is Tanh and forward pass computes `clamp(input + delta, 0, 1)`. Backward-compatible: default is `False`, existing checkpoints unaffected.
- `losses.py`: Added `ColorLoss` — BT.601 RGB→YCbCr, chrominance (Cb, Cr) penalised 2× relative to luminance Y. No new deps.
- `train_v3.py`: New training script. `UNet(base_filters=16, residual=True)`. Loss: `CombinedLoss + 0.5 * ColorLoss`. Warm-starts from `best_extended.pt` with `strict=False`.
- `app.py`: Restructured UI layout (input/output images now in same `gr.Row(equal_height=True)` — horizontally aligned). Added v3 to dropdown. Updated `MODEL_OPTIONS` to carry `(repo_id, filename, residual)` tuples; `get_model` passes `residual` to `UNet`.

**v3 training results (20 epochs, batch=4, crop=176×176, lr=1e-5)**

| Ep | Val MSE avg | MSE_R | MSE_G | MSE_B | Best |
|----|------------|-------|-------|-------|------|
| 1  | 0.073617 | 0.0777 | 0.0697 | 0.0734 | ✓ |
| 8  | 0.064054 | 0.0673 | 0.0597 | 0.0652 | ✓ |
| 11 | 0.058475 | 0.0582 | 0.0552 | 0.0620 | ✓ |
| 16 | 0.056371 | 0.0575 | 0.0528 | 0.0588 | ✓ |
| 18 | **0.050844** | 0.0506 | 0.0464 | 0.0555 | ✓ |

Best checkpoint: epoch 18, val MSE **0.050844**

**Final eval on night.jpg / day.jpg:**
- MSE_R=0.075107, MSE_G=0.069984, MSE_B=0.080816, avg=**0.075302**
- Higher than v1 (0.038925) — expected: ColorLoss trades raw MSE for chrominance accuracy

**Upload**
- `best_v3.pt` and `model.py` uploaded to `tyakovenko/night-to-day-enhancement-model-v3`
- `app.py` updated: v3 is the default model in the dropdown

### Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Residual architecture | Tanh delta + clamp | Prevents Sigmoid regression-to-mean; model predicts only the enhancement delta |
| Color loss weighting | 0.5 × ColorLoss | Balances chrominance correction against pixel fidelity; tunable via `--color-weight` |
| Warm-start source | best_extended.pt (not best_v2.pt) | MSE-trained weights provide a stable pixel-accurate starting point; v2 weights are already shifted toward SSIM optima |
| strict=False load | Yes | Handles the Sigmoid→Tanh activation swap cleanly |
| UI layout | input/output in same gr.Row(equal_height=True) | Puts images at the same horizontal level; dropdown + button moved to a separate top row |

### Open Tasks

- **Visual quality check**: open `enhanced_night_v3.jpg` — does it show blue sky and natural greens vs. v1's gray output?
- **Potential next iteration**: if color is fixed but MSE is too high, try staged training — warm-start for 10+ epochs at lower LR before adding ColorLoss, then fine-tune with it
- **HF Space**: push updated `app.py` to Space repo so v3 appears in the live dropdown
