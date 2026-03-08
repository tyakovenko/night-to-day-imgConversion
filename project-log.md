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
