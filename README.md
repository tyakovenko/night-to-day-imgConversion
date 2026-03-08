# Low-Light to Day Image Enhancement

A U-Net model that enhances low-light and night-time images to match daylight appearance, evaluated by channel-wise MSE.

- **Live demo:** [tyakovenko-night-to-day-enhancement.hf.space](https://tyakovenko-night-to-day-enhancement.hf.space)
- **Dataset:** [tyakovenko/night-to-day-enhancement](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) — 1,176 pairs (Transient Attributes)
- **Extended dataset:** [tyakovenko/night-to-day-enhancement-extended](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement-extended) — adds 500 LOL pairs
- **Model:** [tyakovenko/night-to-day-enhancement-model](https://huggingface.co/tyakovenko/night-to-day-enhancement-model) — best.pt (val MSE 0.0290)
- **Extended model:** [tyakovenko/night-to-day-enhancement-model-extended](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-extended) — best_extended.pt (val MSE 0.0278)

---

## Run the demo locally (Docker)

Anyone with Docker installed can run the Gradio UI locally — no Python setup needed.

```bash
# 1. Clone the repo
git clone https://github.com/tyakovenko/night-to-day-imgConversion.git
cd night-to-day-imgConversion

# 2. Build the image (installs CPU-only torch + all dependencies in a venv)
docker build -t night-to-day .

# 3. Run — UI available at http://localhost:7860
docker run -p 7860:7860 night-to-day
```

On first launch the model checkpoint (~22MB) downloads from HF Hub automatically. Subsequent runs use the Docker layer cache.

> **Requirements:** Docker Desktop (Mac/Windows) or Docker Engine (Linux). No GPU needed.

---

## Reproduce the experiment (Python, no Docker)

### 1. Environment

```bash
git clone https://github.com/tyakovenko/night-to-day-imgConversion.git
cd night-to-day-imgConversion

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Train the baseline model (Transient Attributes dataset)

Training streams images from HF Hub on first run (~1,266 files, cached locally after).

```bash
python train.py \
  --epochs 30 \
  --batch-size 8 \
  --crop-size 128 \
  --base-filters 16 \
  --lr 1e-4 \
  --workers 4
```

Expected outcome: `checkpoints/best.pt` at ~epoch 22, val MSE avg ≈ **0.029**.
Runtime: ~90–110s/epoch on CPU.

### 3. Train the extended model (Transient Attributes + LOL)

Requires the [LOL dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) downloaded locally. Place it so `low/` and `high/` subfolders are accessible, then run the upload script once to push LOL images to HF Hub and extend the manifest:

```bash
python - <<'EOF'
import shutil
from pathlib import Path
from huggingface_hub import upload_large_folder

LOL_SRC = Path("/path/to/lol_dataset")   # <-- update this path
STAGE   = Path("/tmp/lol_hf_stage")

for src, dst in [("low", "low_light/lol"), ("high", "day/lol")]:
    (STAGE / dst).mkdir(parents=True, exist_ok=True)
    for f in (LOL_SRC / src).iterdir():
        shutil.copy2(f, STAGE / dst / f.name)

upload_large_folder(
    repo_id="tyakovenko/night-to-day-enhancement-extended",
    repo_type="dataset",
    folder_path=str(STAGE),
)
EOF
```

Then fine-tune from the baseline checkpoint:

```bash
python train_extended.py \
  --epochs 20 \
  --batch-size 8 \
  --crop-size 128 \
  --lr 1e-5 \
  --init-checkpoint checkpoints/best.pt \
  --workers 4
```

Expected outcome: `checkpoints/best_extended.pt`, val MSE avg ≈ **0.028**, improving on the baseline.

### 4. Run inference on a single image

```bash
python enhance.py --input night.jpg --output enhanced_night.jpg --checkpoint checkpoints/best.pt
```

### 5. Evaluate against a ground-truth reference

```bash
python enhance.py --input night.jpg --reference day.jpg --checkpoint checkpoints/best.pt
```

Output:
```
MSE_R:   0.037988
MSE_G:   0.035524
MSE_B:   0.043262
MSE_avg: 0.038925
```

---

## Results

| Model | Val MSE avg | MSE R | MSE G | MSE B | Best epoch |
|---|---|---|---|---|---|
| Baseline (Transient Attributes) | 0.028953 | 0.028700 | 0.025959 | 0.032200 | 22 |
| Extended (+ LOL fine-tune) | 0.027752 | 0.026945 | 0.025522 | 0.030788 | 19 |

**Final evaluation on held-out pair `night.jpg` / `day.jpg` (1024×737):**

| MSE_R | MSE_G | MSE_B | MSE_avg |
|---|---|---|---|
| 0.037988 | 0.035524 | 0.043262 | 0.038925 |

---

## Architecture

4-level U-Net (`base_filters=16`) trained with MSE loss. Input images are padded to the nearest multiple of 16 before inference (reflect padding, bottom/right only) and cropped back to original dimensions — handles any input size without resizing.

```
Input (H×W×3)
  → Encoder: 4× [ConvBlock + MaxPool2d]
  → Bottleneck: ConvBlock
  → Decoder: 4× [Upsample + skip connection + ConvBlock]
  → Output (H×W×3, Sigmoid)
```

Training details: crop size 128×128, batch 8, Adam optimizer, `ReduceLROnPlateau` scheduler (factor=0.5, patience=5). Scene-level 85/15 train/val split to prevent data leakage.

---

## Repository structure

| File | Purpose |
|---|---|
| `model.py` | U-Net architecture |
| `dataset.py` | Dataset class; streams pairs from HF Hub via manifest CSV |
| `train.py` | Baseline training loop |
| `train_extended.py` | Fine-tuning on combined TA + LOL dataset |
| `enhance.py` | Single-image inference + MSE evaluation |
| `app.py` | Gradio UI (also entry point for Docker) |
| `Dockerfile` | Docker build: Python 3.11-slim, venv, CPU-only torch |
| `low_light_manifest.csv` | 1,176 TA training pairs |
| `extended_manifest.csv` | 1,676 combined pairs (TA + LOL) |
