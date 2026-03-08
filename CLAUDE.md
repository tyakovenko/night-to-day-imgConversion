# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CLAUDE.md — Low-Light to Day Image Enhancement

## Commands

```bash
# Activate environment (always do this first)
source venv/bin/activate

# Install / sync dependencies
pip install -r requirements.txt

# Re-run data analysis (Phase 1 — already complete, only if manifest needs regeneration)
python analyze_dataset.py

# Re-upload dataset to HF Hub (use v3 only — v1/v2 are broken for bulk uploads)
python upload_to_hf_v3.py

# Train model
python train.py

# Run inference on a single image
python enhance.py --input night.jpg --output enhanced_night.jpg

# Evaluate against ground truth (channel-wise MSE) — final eval pair
python enhance.py --input night.jpg --reference day.jpg

# Launch Gradio demo locally
python app.py
```

---

## Architecture

### Phase Status

- **Phase 1 (Data Analysis):** COMPLETE — `low_light_manifest.csv` + `data-analyst-report.md` generated; dataset on HF Hub
- **Phase 2 (Model + Training pipeline):** COMPLETE — U-Net trained (epoch 22, val MSE 0.028953); `best.pt` on HF Hub
- **Phase 3 (UI / deployment):** IN PROGRESS — Space live; inference padding bug still needs fix in `app.py`

### Data Flow

```
HF Hub: tyakovenko/night-to-day-enhancement
    ├── low_light_manifest.csv   ← sole authoritative source of (low-light, day) pairs
    ├── low_light/<scene>/<img>  ← input images
    └── day/<scene>/<img>        ← target images

low_light_manifest.csv columns:
    scene              — scene folder ID (e.g. "00000064")
    low_light_image    — relative path, e.g. "00000064/3.jpg"
    day_target_image   — relative path, e.g. "00000064/12.jpg"
    low_light_reason   — which thresholds triggered classification
```

All code that loads training data must stream images from HF Hub using `low_light_manifest.csv`. Never reference local paths (`transientAttributesDataset/` or `hf_staging/`).

### Scripts (existing)

| Script | Purpose |
|--------|---------|
| `analyze_dataset.py` | Parses annotations.tsv, selects day/low-light pairs, writes manifest + report |
| `upload_to_hf_v3.py` | Uploads images + manifest to HF Hub via `upload_large_folder()` |
| `upload_to_hf.py` / `upload_to_hf_v2.py` | Deprecated upload attempts — do not use |

### Dataset Class Design

Must be dataset-agnostic: Transient Attributes and LOL datasets share the same `(low_light_image, day_target_image)` schema. LOL can be appended to the manifest with `low_light_reason="lol_dataset"`.

---

## Project

**Name:** Low-Light to Day Image Enhancement
**Description:** Enhance low-light images (including night-time) to match day-time appearance, evaluated by channel-wise MSE.  

## Stack

**Language(s):** Python 3.11  
**Framework(s):** PyTorch, torchvision, OpenCV, NumPy, Pillow, scikit-image, pandas, gradio  
**Database:** Hugging Face Hub (dataset hosting + model checkpoints)  
**Dependency isolation:** venv  

## Deployment

**Target:** local (training) + Hugging Face Spaces (Gradio demo + inference)  
**HF Spaces:** `huggingface.co/spaces/tyakovenko/night-to-day-enhancement` 

## UI Design

**Platform:** Gradio on Hugging Face Spaces  
- Input: low-light image upload  
- Output: enhanced image + channel-wise MSE (if day reference provided)  

## Agent Configuration

**Active agents:** DataAnalyst, Lead, Planner, Architect, Backend, Frontend, QA, Scribe  
**Skip agents:** Security

### Lead — Session Flow

**Phase 1 — Data Analysis (blocking)**

Lead checks for Phase 1 completion before doing anything else. Phase 1 is **already complete** if all three of the following are true:

- `data-analyst-report.md` exists and is non-empty
- `low_light_manifest.csv` exists and is non-empty
- `hf_upload_status.txt` contains `status: SUCCESS`

If all three conditions are met, **skip DataAnalyst entirely** and proceed directly to Phase 2.

If any condition is not met, Lead spawns **DataAnalyst** and passes it:

- `transientAttributesDataset/annotations.tsv`
- `transientAttributesDataset/imageAlignedLD/`
- `agents/data-analyst.md` (the agent's own spec)

Lead waits for DataAnalyst to return. DataAnalyst is considered done when all three conditions above are satisfied.

**No other agent is spawned and no other task begins until Phase 1 is complete.**

**Phase 2 — Downstream work**

Once `low_light_manifest.csv` exists, Lead resumes normal orchestration. All agents that build or train on image pairs must read their pairs exclusively from `low_light_manifest.csv` — never from raw annotations or their own re-derived logic.

---

## Project-Specific Rules

### Hard Constraints

- **Inference integrity:** At test time, only the low-light image is input. Day image may be used for training/supervision/parameter selection — never copied or blended at inference.
- **Generalizability:** Pipeline must not overfit to any single pair. Hidden test pairs may be used for grading.
- **Pixel precision:** No accidental cropping, resizing, or padding. Output must match reference dimensions exactly.

### Evaluation Metric

Channel-wise MSE: compute MSE per channel (R, G, B), then average. Report all four values.  
Compute in **float64** to avoid accumulation errors. Normalize consistently (0–255 or 0–1).

### Dataset

**Training data:** Transient Attributes Dataset — http://transattr.cs.brown.edu/  
**Access via:** Hugging Face Hub — `tyakovenko/night-to-day-enhancement` (uploaded, 2,537 files, 255.6 MB)

Dataset structure: 102 scene directories in `imageAlignedLD/`, each containing time-of-day captures of the same scene. Annotations: 40 attributes per image (score + confidence) in `annotations.tsv`.

**Pair selection strategy — delegated to DataAnalyst:**

All low-light classification and pair selection is performed by the **DataAnalyst** agent (see [`agents/data-analyst.md`](agents/data-analyst.md)). The output of that analysis is `low_light_manifest.csv`, which is the **sole authoritative source** of training pairs. Backend reads this file directly — do not re-derive pairs from annotations.

Summary of DataAnalyst's approach (details in report):
- **Day targets:** `daylight > 0.8` per scene (DataAnalyst validates and may adjust)
- **Low-light inputs:** determined by full analysis of all 40 attributes, not just `night`. Thresholds are chosen with explicit distribution-based justification.
- **Many-to-one mapping:** multiple low-light images per scene may map to one day target.
- **Excluded scenes:** any scene missing a valid daylight or low-light image is dropped and documented.

**Final evaluation pair (in repo root):** `night.jpg` (input) / `day.jpg` (ground truth) — 1024×737. These are used for a **single final evaluation run only** — never for training or model testing. The model's grade is determined by channel-wise MSE on this pair. Additional hidden pairs may also be used.

**Eval pair note:** Height 737 is not divisible by 16. `enhance.py` handles this with reflect padding (see `pad_to_multiple`). If hidden eval pairs have similar non-multiple-of-16 dimensions, the same code path handles them automatically.

**Data notes:**
- Some night images are very noisy in low-light — expected
- Images within same scene folder are spatially aligned

### Extension Plan: LOL Dataset

LOL (https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) — 485 paired low/normal-light images.  
Design the `Dataset` class to be dataset-agnostic so LOL plugs in as a second training phase without code changes.  
Risk: LOL is indoor — fine-tune only after Transient Attributes training, or weight lower in mixed training.

### Coding Conventions

- Explicitly handle BGR (OpenCV) vs RGB (PIL/torchvision) at every I/O boundary
- Reproducible seeds: `torch.manual_seed`, `numpy.random.seed`, `torch.use_deterministic_algorithms(True)`
- Separate scripts: `train.py`, `enhance.py` (inference), `app.py` (Gradio)
- All hyperparameters via `argparse` or config — no hardcoded values
- Log every experiment: method, hyperparams, per-channel MSE, avg MSE, runtime
- Cite all third-party code with `# SOURCE: <url>` comments

## Session State
# Maintained automatically by Scribe in ./project-log.md
