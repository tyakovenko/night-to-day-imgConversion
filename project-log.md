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

## Session: 2026-03-08

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

### Training Results (in progress — 30 epochs total)

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

**Best so far:** Val MSE 0.032454 at epoch 8 (checkpoint: `checkpoints/best.pt`)
**Observation:** Blue channel (MSE_B) consistently highest — night scenes suppress blue/cool tones most.

### Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | U-Net (base_filters=16) | Proven for pixel-wise image translation; skip connections preserve spatial detail |
| Loss | MSE (L2) | Directly optimizes evaluation metric |
| Train/val split | Scene-level 85/15 | Prevents data leakage between train and test |
| Crop size | 128×128 | CPU-feasible (~90s/epoch); full resolution used at inference |
| Initial config (killed) | base_filters=32, crop=256 | ~20 min/epoch on CPU — too slow |

### Open Tasks

- Wait for training to complete (30 epochs, ~19 epochs remaining)
- Run `enhance.py --input night.png --reference day.png` on final evaluation pair
- Implement `app.py` Gradio demo
- Upload model checkpoint to HF Hub
