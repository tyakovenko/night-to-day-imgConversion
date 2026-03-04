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

**Active agents:** Lead, Planner, Architect, Backend, Frontend, QA, Scribe  
**Skip agents:** Security  

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

**Pair selection strategy — many-to-one mapping:**
- **Day targets:** `daylight > 0.8` from each scene
- **Low-light inputs (not strictly night):** The training inputs are **low-light images** — a broader category than night-time. Always include `night > 0.8` images, but also analyze all 40 annotations to identify other low-visibility conditions. Suggested attributes: `dark`, `dawndusk`, `sunrisesunset`, `gloomy`, `storm`, `fog` — but do not limit to these. Use annotation distributions to find the best thresholds.
- **Map multiple low-light images → one day target** per scene to improve generalization.
- Check `annotations.tsv` attribute distributions to calibrate thresholds.
- **Document thoroughly:** Record every data preparation step — attribute analysis, threshold selection rationale, which images were included/excluded, and the final input→output pair counts per scene. This documentation should be reproducible so anyone can reconstruct the exact dataset from the raw annotations.

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

### Academic Integrity

All code original or clearly credited. Third-party models cited in code headers.

---

## Session State
# Maintained automatically by Scribe in ./project-log.md
