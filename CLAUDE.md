# CLAUDE.md — Low-Light to Day Image Enhancement

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
**HF Spaces:** `huggingface.co/spaces/tyakovenko/night-to-day-enhancement` (to be created)  

## UI Design

**Platform:** Gradio on Hugging Face Spaces  
- Input: low-light image upload  
- Output: enhanced image + channel-wise MSE (if day reference provided)  

## Agent Configuration

**Active agents:** DataAnalyst, Lead, Planner, Architect, Backend, Frontend, QA, Scribe  
**Skip agents:** Security

### Lead — Session Flow

**Phase 1 — Data Analysis (blocking)**

Lead spawns **DataAnalyst** as the first action of every session. Lead passes it:

- Path to `annotations.tsv`
- Path to the `imageAlignedLD/` scene directories
- Path to `agents/data-analyst.md` (the agent's own spec)

Lead waits for DataAnalyst to return. DataAnalyst is considered done when both of the following exist and are non-empty:

- `data-analyst-report.md`
- `low_light_manifest.csv`

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
**Access via:** Hugging Face Hub (to be uploaded as `tyakovenko/transient-attributes-day-night`)

Dataset structure: 102 scene directories in `imageAlignedLD/`, each containing time-of-day captures of the same scene. Annotations: 40 attributes per image (score + confidence) in `annotations.tsv`.

**Pair selection strategy — delegated to DataAnalyst:**

All low-light classification and pair selection is performed by the **DataAnalyst** agent (see [`agents/data-analyst.md`](agents/data-analyst.md)). The output of that analysis is `low_light_manifest.csv`, which is the **sole authoritative source** of training pairs. Backend reads this file directly — do not re-derive pairs from annotations.

Summary of DataAnalyst's approach (details in report):
- **Day targets:** `daylight > 0.8` per scene (DataAnalyst validates and may adjust)
- **Low-light inputs:** determined by full analysis of all 40 attributes, not just `night`. Thresholds are chosen with explicit distribution-based justification.
- **Many-to-one mapping:** multiple low-light images per scene may map to one day target.
- **Excluded scenes:** any scene missing a valid daylight or low-light image is dropped and documented.

**Final evaluation pair (in repo root):** `night.png` (input) / `day.png` (ground truth). These are used for a **single final evaluation run only** — never for training or model testing. The model's grade is determined by channel-wise MSE on this pair. Additional hidden pairs may also be used.

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
