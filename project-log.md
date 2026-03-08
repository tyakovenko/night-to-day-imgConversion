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

### Open Tasks

- Upload `checkpoints/best.pt` to HF Hub model repo
- Connect model to `app.py` (flip `MODEL_LOADED=True`, wire checkpoint loader)
- Run `enhance.py --input night.png --reference day.png` on final evaluation pair
- Fix HF Space: lightweight `requirements.txt` (remove torch until model wired) → rebuild
