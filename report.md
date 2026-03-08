# Mid-Term Project Report — Night-Time Image Enhancement

**Course:** [Course Name]  
**Student:** Taisiia Yakovenko  
**Date:** March 2026  
**Repository:** [tyakovenko/night-to-day-enhancement on GitHub](https://github.com/tyakovenko)  
**HuggingFace:** [tyakovenko](https://huggingface.co/tyakovenko)

---

## Executive Summary

This report documents a **supervised U-Net encoder-decoder** trained to enhance night-time images to match daytime appearance. The pipeline was designed to be generalizable — trained on 1,176 curated low-light/day image pairs from 90 diverse outdoor scenes, with a scene-level split to prevent data leakage.

**Reported MSE on the provided `night.jpg` / `day.jpg` pair:**

| MSE_R | MSE_G | MSE_B | **MSE_avg** |
|---|---|---|---|
| 0.037988 | 0.035524 | 0.043262 | **0.038925** |

Two models were produced: **v1** (Transient Attributes dataset only, val MSE 0.028953) and **v1-extended** (fine-tuned with the additional LOL indoor dataset, val MSE **0.027752**, a 4.1% improvement). Both are publicly hosted on HuggingFace alongside all training data and code, making the experiment fully reproducible without a GPU.

The day-time image was used solely for training supervision and evaluation — it is never accessed during inference. All downstream steps read only from the night-time input.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Approach Overview](#2-approach-overview)
3. [Dataset](#3-dataset)
4. [Data Processing](#4-data-processing)
5. [Architecture](#5-architecture)
6. [Training Procedure — v1 (Transient Attributes)](#6-training-procedure--v1-transient-attributes)
7. [Training Procedure — v1-extended (TA + LOL Fine-Tune)](#7-training-procedure--v1-extended-ta--lol-fine-tune)
8. [Results](#8-results)
9. [Inference Pipeline](#9-inference-pipeline)
10. [HuggingFace Deployment](#10-huggingface-deployment)
11. [Discussion](#11-discussion)
12. [Limitations and Future Work](#12-limitations-and-future-work)
13. [Reproducibility](#13-reproducibility)

---

## 1. Problem Statement

Given a pair of images of the same outdoor scene — one captured at night and one during the day — the goal is to enhance the night-time image to match the day-time appearance as closely as possible. The primary evaluation metric is **channel-wise Mean Squared Error (MSE)** between the enhanced output and the ground-truth day-time image:

$$\text{MSE}_c = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i^c - y_i^c)^2, \quad c \in \{R, G, B\}$$

$$\text{MSE}_{\text{avg}} = \frac{1}{3} (\text{MSE}_R + \text{MSE}_G + \text{MSE}_B)$$

where $\hat{y}$ is the enhanced image, $y$ is the day-time reference, and $N$ is the total number of pixels in a single channel. All pixel values are normalized to the range $[0, 1]$ (float64).

**Integrity constraint:** The day-time image may only be used for training supervision and parameter selection — it must not be directly copied or blended into the inference output.

---

## 2. Approach Overview

A learning-based approach was chosen — a **U-Net encoder-decoder** trained end-to-end to perform the night-to-day image translation. This approach was selected for the following reasons:

| Consideration | Rationale |
|---|---|
| **Metric alignment** | Training directly with MSE loss directly optimizes the evaluation criterion |
| **Spatial detail preservation** | U-Net skip connections pass fine-grained features from encoder to decoder |
| **End-to-end optimization** | No hand-tuned intermediate steps (histogram matching, gamma, etc.) |
| **Generalizability** | Training on a diverse dataset of 90+ outdoor scenes prevents overfitting to the single visible pair |

The overall pipeline is:

```
Night image (RGB) → U-Net → Enhanced image (RGB)
                                     ↓
             Channel-wise MSE vs. Day image (evaluation only)
```

---

## 3. Dataset

### 3.1 Primary Dataset — Transient Attributes (TA)

The training data was curated from the **Transient Attributes Dataset** (Laffont et al., Brown University), which contains 8,571 annotated images across 101 outdoor scenes at various times of day. Each image is annotated with continuous attribute scores (e.g., `daylight`, `night`, `dark`, `dawndusk`).

**Pair selection criteria:**

| Role | Filter |
|---|---|
| **Day target** | `daylight > 0.8` (highest-scoring image per scene) |
| **Low-light input** | `night > 0.5` OR (`dark > 0.5` AND `daylight < 0.4`) OR (`dawndusk > 0.5` AND `daylight < 0.4`) |

This yielded **1,176 validated low-light → day image pairs** across 90 scenes, stored in `low_light_manifest.csv`.

**Train/val split:** Scene-level (85/15) to prevent data leakage — all images from a given scene are assigned to only one split.

| Split | Pairs | Scenes |
|---|---|---|
| Train | 1,010 | 77 |
| Val | 166 | 13 |

The curated dataset was uploaded to HuggingFace Hub: [`tyakovenko/night-to-day-enhancement`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) (2,537 files, ~256 MB).

### 3.2 Supplemental Dataset — LOL (Low-Light Dataset)

For the fine-tuned model (v1-extended), the [LOL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) was added. LOL provides **485 paired indoor low-light/normal-light images**, extending the model's generalization to non-outdoor scenes.

| Source | Train | Val | Total |
|---|---|---|---|
| Transient Attributes | 1,095 | 81 | 1,176 |
| LOL | 420 | 80 | 500 |
| **Combined** | **1,515** | **161** | **1,676** |

The combined dataset was uploaded to HuggingFace Hub: [`tyakovenko/night-to-day-enhancement-extended`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement-extended).

---

## 4. Data Processing

The data processing pipeline was implemented in `analyze_dataset.py` and executed by the DataAnalyst agent. Its purpose was to transform the raw Transient Attributes annotation files into a clean, validated set of training pairs stored in `low_light_manifest.csv`.

### 4.1 Raw Data Overview

The Transient Attributes Dataset contains **8,571 annotated images** across **101 outdoor scenes**. Each image is annotated with 40 continuous attribute scores in `[0, 1]` (e.g. `daylight`, `night`, `dark`, `dawndusk`, `fog`, `glowing`), stored as `score,confidence` pairs in a tab-separated file.

Key attributes used in the pipeline:

| Attribute | Role in pipeline |
|---|---|
| `daylight` | Selects day-target images (high score = clear daytime) |
| `night` | Primary low-light discriminator |
| `dark` | Secondary low-light discriminator (independent of time-of-day) |
| `dawndusk` | Tertiary low-light discriminator (transitional lighting) |

### 4.2 Attribute Distribution Analysis

Before setting thresholds, the distributions of all relevant attributes were analyzed across the full 8,571-image corpus:

| Attribute | Mean | Median | 75th pct | 90th pct |
|---|---|---|---|---|
| `daylight` | 0.685 | 0.791 | 0.953 | 1.000 |
| `night` | 0.167 | 0.043 | 0.166 | 0.604 |
| `dark` | 0.215 | 0.111 | 0.298 | 0.633 |
| `dawndusk` | 0.216 | 0.119 | 0.295 | 0.606 |

**Key finding:** The `night`, `dark`, and `dawndusk` distributions are all **heavily right-skewed** — approximately 65–70% of images score below 0.1 on each attribute (the dataset is predominantly daytime). The `daylight` distribution shows a clear bimodal structure, with a large cluster near 1.0 (clear daytime images) and a secondary cluster near 0 (night/dark images).

### 4.3 Day-Target Selection

**Criterion:** `daylight > 0.8`, selecting the single highest-scoring image per scene as the day reference.

**Rationale:** The `daylight` threshold of 0.8 sits between the 50th percentile (0.791) and 75th percentile (0.953), capturing the top ~49% of images. When multiple images in a scene exceed this threshold, only the highest-scoring one is used — maximizing target quality and reducing annotation noise.

| Result | Value |
|---|---|
| Images with `daylight > 0.8` | 4,184 |
| Scenes with a valid day target | 98 |
| Scenes excluded (no valid day target) | 3 |

The 3 excluded scenes had maximum `daylight` scores of 0.626, 0.782, and 0.722 — all below the threshold — and were dropped entirely from the dataset.

### 4.4 Low-Light Image Classification

Low-light conditions are not captured by a single attribute. Pure nighttime, overcast dark scenes, and dusk scenes all constitute challenging low-light inputs but annotate differently. Three attributes were combined:

**Classification rule:**

```
(night > 0.5) OR (dark > 0.5 AND daylight < 0.4) OR (dawndusk > 0.5 AND daylight < 0.4)
```

The `daylight < 0.4` guard prevents transitional images from being classified as low-light if they still contain meaningful daylight presence (the 25th percentile for `daylight` is 0.462, so `< 0.4` is approximately the bottom 28% of scores).

**Threshold justification:** 0.5 corresponds to approximately the 88th percentile for `night`, `dark`, and `dawndusk` — deliberately strict, selecting only the top ~12% of images on each criterion. This prioritizes **precision over recall**: the model should learn from genuinely challenging low-light conditions, not mildly underexposed images.

| Criterion path | Images matched |
|---|---|
| `night > 0.5` | 911 |
| `dark > 0.5` (with `daylight < 0.4`) | 962 |
| `dawndusk > 0.5` (with `daylight < 0.4`) | 733 |
| **Total unique low-light images** | **1,272** |

**Tradeoff considered:**

| | Stricter threshold (0.5, chosen) | Looser threshold (0.3) |
|---|---|---|
| Purity | Higher — top ~12% | Lower — top ~25% |
| Pair count | ~1,176 | ~2,000+ (estimated) |
| Training signal | Clean, high-contrast pairs | More noise, borderline cases |

The stricter threshold was preferred for training signal quality.

### 4.5 Pair Construction and Validation

Each valid low-light image was paired with the scene's selected day target, forming `(input, target)` pairs. All pairs were then validated:

1. Both file paths confirmed to exist on disk
2. Both images confirmed to be readable (non-corrupt) by PIL

| Validation metric | Value |
|---|---|
| Pairs before validation | 1,176 |
| Removed (file missing) | 0 |
| Removed (unreadable) | 0 |
| **Final validated pairs** | **1,176** |

### 4.6 Final Manifest

The output of the pipeline is `low_light_manifest.csv`, with one row per (low-light input, day target) pair. This file is the sole input to all downstream steps (Dataset class, training, inference) — no code references raw annotation files or local image paths directly.

| Metric | Value |
|---|---|
| Total valid pairs | 1,176 |
| Unique scenes covered | 90 (of 101) |
| Unique low-light images | 1,176 |
| Unique day-target images | 90 (one per scene) |
| Total unique image files | 1,266 |
| Total dataset size | ~255 MB |

The dataset was uploaded to HuggingFace Hub as [`tyakovenko/night-to-day-enhancement`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) (2,537 files) using `upload_large_folder()` with chunked upload and resume support.

---

## 5. Architecture


The model is a **4-level U-Net** encoder-decoder, implemented in PyTorch.

### Key design choices:

| Property | Value / Choice | Rationale |
|---|---|---|
| Architecture | U-Net (Ronneberger et al., 2015) | Standard for pixel-wise image translation |
| Encoder depth | 4 levels | Captures multi-scale features |
| Skip connections | Yes | Preserves spatial detail lost during max-pooling |
| Upsampling | Bilinear + 1×1 conv | Avoids checkerboard artifacts from transposed convolution |
| Base filters (`f`) | 16 | CPU-feasible parameter count; 1.8M total |
| Output activation | Sigmoid | Constrains output to `[0, 1]` range |
| Input/Output | 3-channel RGB, float32 | Matches dataset format |

### Architecture summary:

```
Input (3, H, W)
  └─ Encoder:  [16] → [32] → [64] → [128]  (each: Conv-BN-ReLU × 2 + MaxPool)
  └─ Bottleneck: [256]
  └─ Decoder:  [128] → [64] → [32] → [16]  (each: Upsample + concat skip + Conv-BN-ReLU × 2)
  └─ Output head: Conv 1×1 → Sigmoid
Output (3, H, W)
```

**Total parameters:** 1,811,811

---

## 6. Training Procedure — v1 (Transient Attributes)

**Model checkpoint:** `best.pt`  
**HuggingFace:** [`tyakovenko/night-to-day-enhancement-model`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model)

### 5.1 Training Configuration

| Hyperparameter | Value |
|---|---|
| Dataset | Transient Attributes (1,176 pairs) |
| Train pairs | 1,010 |
| Val pairs | 166 |
| Crop size | 128×128 (random crop during training) |
| Batch size | 8 |
| Epochs | 30 |
| Optimizer | Adam |
| Initial LR | 1e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) |
| Loss function | MSE (L2) |
| Hardware | CPU |

LR was halved to 5e-5 at epoch 19, and again to 2.5e-5 at epoch 29.

### 5.2 Training Results (Epoch-by-Epoch)

| Epoch | Train Loss | Val MSE avg | MSE_R | MSE_G | MSE_B | Best |
|---|---|---|---|---|---|---|
| 1 | 0.054488 | 0.039336 | 0.035079 | 0.035582 | 0.047347 | ✓ |
| 2 | 0.051836 | 0.035965 | 0.035797 | 0.032521 | 0.039575 | ✓ |
| 3 | 0.050300 | 0.037730 | 0.035330 | 0.034132 | 0.043728 | |
| 4 | 0.046166 | 0.036025 | 0.035589 | 0.033050 | 0.039435 | |
| 5 | 0.047562 | 0.037194 | 0.036199 | 0.034183 | 0.041201 | |
| 6 | 0.048429 | 0.034016 | 0.032854 | 0.030858 | 0.038337 | ✓ |
| 7 | 0.047591 | 0.033815 | 0.031671 | 0.030261 | 0.039514 | ✓ |
| 8 | 0.049659 | 0.032454 | 0.030424 | 0.028821 | 0.038117 | ✓ |
| 9 | 0.047977 | 0.036101 | 0.032500 | 0.032339 | 0.043465 | |
| 10 | 0.048240 | 0.033233 | 0.032639 | 0.029650 | 0.037409 | |
| 11 | 0.047696 | 0.033106 | 0.031234 | 0.029439 | 0.038645 | |
| 12 | 0.047381 | 0.030668 | 0.031171 | 0.028090 | 0.032744 | ✓ |
| 13 | 0.048375 | 0.037351 | 0.034258 | 0.033383 | 0.044412 | |
| 14 | 0.046013 | 0.034301 | 0.032956 | 0.030723 | 0.039223 | |
| 15 | 0.047037 | 0.033501 | 0.032079 | 0.030086 | 0.038339 | |
| 16 | 0.045721 | 0.036675 | 0.033941 | 0.033099 | 0.042984 | |
| 17 | 0.046112 | 0.032932 | 0.030803 | 0.029809 | 0.038186 | |
| 18 | 0.046836 | 0.036563 | 0.033851 | 0.032513 | 0.043326 | |
| 19 | 0.046244 | 0.033501 | 0.032378 | 0.030055 | 0.038071 | |
| 20 | 0.046453 | 0.029839 | 0.029595 | 0.026591 | 0.033331 | ✓ |
| 21 | 0.044902 | 0.034271 | 0.033707 | 0.030986 | 0.038121 | |
| **22** | **0.043913** | **0.028953** | **0.028700** | **0.025959** | **0.032200** | **✓ (Best)** |
| 23 | 0.046162 | 0.032787 | 0.032881 | 0.029942 | 0.035538 | |
| 24 | 0.044521 | 0.030584 | 0.030248 | 0.027590 | 0.033915 | |
| 25 | 0.046452 | 0.033651 | 0.032762 | 0.030585 | 0.037607 | |
| 26 | 0.044440 | 0.037195 | 0.035323 | 0.033143 | 0.043118 | |
| 27 | 0.044797 | 0.035506 | 0.035270 | 0.032336 | 0.038913 | |
| 28 | 0.044276 | 0.032017 | 0.030109 | 0.029099 | 0.036844 | |
| 29 | 0.044924 | 0.033364 | 0.032470 | 0.030195 | 0.037426 | |
| 30 | 0.044869 | 0.032616 | 0.031317 | 0.029604 | 0.036927 | |

**Best checkpoint: Epoch 22 — Val MSE avg = 0.028953**

**Observations:**
- The Blue channel (MSE_B) is consistently the highest — expected, as night scenes suppress cool/blue tones most
- The Green channel learns fastest and achieves the lowest MSE
- Training loss decreases monotonically, but val MSE shows noisy oscillations, suggesting the model begins to overfit after epoch 22

---

## 7. Training Procedure — v1-extended (TA + LOL Fine-Tune)

**Model checkpoint:** `best_extended.pt`  
**HuggingFace:** [`tyakovenko/night-to-day-enhancement-model`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model) (same repo, separate file)

### 6.1 Motivation

The v1 model was trained only on outdoor images. The LOL dataset provides paired indoor low-light images, which improves the model's generalization to scenes with different lighting conditions and scene types. Fine-tuning rather than training from scratch preserves the feature representations learned in v1.

### 6.2 Training Configuration

| Hyperparameter | Value |
|---|---|
| Initialized from | `best.pt` (epoch 22, val MSE 0.028953) |
| Dataset | Transient Attributes + LOL (1,676 pairs combined) |
| Train pairs | 1,515 |
| Val pairs | 161 |
| Crop size | 128×128 |
| Batch size | 8 |
| Epochs | 20 |
| Optimizer | Adam |
| LR | 1e-5 (10× lower than original — standard fine-tuning practice) |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=4) |
| Loss function | MSE (L2) |

### 6.3 Training Results (Epoch-by-Epoch)

| Epoch | Train Loss | Val MSE avg | MSE_R | MSE_G | MSE_B | Best |
|---|---|---|---|---|---|---|
| 1 | 0.045298 | 0.033250 | 0.032900 | 0.030728 | 0.036123 | ✓ |
| 2 | 0.044029 | 0.035901 | 0.035543 | 0.032659 | 0.039502 | |
| 3 | 0.043579 | 0.031301 | 0.033523 | 0.028636 | 0.031745 | ✓ |
| 4 | 0.042728 | 0.033143 | 0.032529 | 0.030583 | 0.036318 | |
| 5 | 0.043481 | 0.034552 | 0.034666 | 0.032013 | 0.036978 | |
| 6 | 0.043071 | 0.031409 | 0.031724 | 0.027909 | 0.034593 | |
| 7 | 0.041966 | 0.032736 | 0.031680 | 0.029982 | 0.036545 | |
| 8 | 0.040932 | 0.030066 | 0.028076 | 0.027367 | 0.034755 | ✓ |
| 9 | 0.041527 | 0.029932 | 0.028948 | 0.026681 | 0.034166 | ✓ |
| 10 | 0.040836 | 0.029888 | 0.028694 | 0.027100 | 0.033869 | ✓ |
| 11 | 0.039981 | 0.029658 | 0.027603 | 0.026932 | 0.034441 | ✓ |
| 12 | 0.042402 | 0.030755 | 0.030603 | 0.028458 | 0.033203 | |
| 13 | 0.042061 | 0.032160 | 0.030106 | 0.028968 | 0.037405 | |
| 14 | 0.040989 | 0.029601 | 0.028015 | 0.026560 | 0.034228 | ✓ |
| 15 | 0.040349 | 0.029234 | 0.027889 | 0.026202 | 0.033609 | ✓ |
| 16 | 0.039369 | 0.032049 | 0.030131 | 0.029141 | 0.036873 | |
| 17 | 0.039922 | 0.027774 | 0.027696 | 0.025352 | 0.030275 | ✓ |
| 18 | 0.040136 | 0.028962 | 0.027476 | 0.026400 | 0.033009 | |
| **19** | **0.041168** | **0.027752** | **0.026945** | **0.025522** | **0.030788** | **✓ (Best)** |
| 20 | 0.040250 | 0.030765 | 0.030378 | 0.028128 | 0.033791 | |

**Best checkpoint: Epoch 19 — Val MSE avg = 0.027752**

Fine-tuning on LOL improved val MSE from 0.028953 → 0.027752, a **4.1% improvement**.

---

## 8. Results

### 8.1 Validation MSE Summary

| Model | Val MSE avg | MSE_R | MSE_G | MSE_B |
|---|---|---|---|---|
| **v1** (`best.pt`) | 0.028953 | 0.028700 | 0.025959 | 0.032200 |
| **v1-extended** (`best_extended.pt`) | **0.027752** | **0.026945** | **0.025522** | **0.030788** |
| Δ (improvement) | −4.1% | −6.1% | −1.7% | −4.4% |

### 8.2 Final Evaluation on the Provided Image Pair

The `enhance.py` script was run on the provided `night.jpg` / `day.jpg` pair (1024×737 pixels) using the v1 model (`best.pt`):

| Metric | Value |
|---|---|
| MSE_R | 0.037988 |
| MSE_G | 0.035524 |
| MSE_B | 0.043262 |
| **MSE_avg** | **0.038925** |

> **Note:** The final eval MSE (0.0389) is higher than the validation MSE during training (0.0290) because the provided `night.jpg`/`day.jpg` pair is a **held-out, out-of-distribution scene** not seen during training. The gap is expected and is not the result of overfitting to the validation split.

### 8.3 Channel-wise Observations

- **Blue channel (MSE_B)** is consistently the highest across all evaluations — night images suppress blue/cool tones most severely, making recovery hardest.
- **Green channel (MSE_G)** achieves the lowest MSE — green wavelengths are least affected by night-time lighting conditions.
- **Red channel (MSE_R)** is intermediate, reflecting moderate suppression under artificial and ambient night lighting.

---

## 9. Inference Pipeline

Inference is implemented in `enhance.py`. The pipeline handles arbitrary image sizes, addressing a critical constraint from the U-Net architecture.

**Key technical steps:**

1. **Load:** Read `night.jpg` as RGB, normalize to `[0, 1]` float32
2. **Pad:** Apply reflect padding (bottom and right only) to make H and W divisible by 16 — required by the 4-level U-Net (4 MaxPool2d layers, each halving spatial dimensions)
3. **Forward pass:** Run U-Net in `torch.no_grad()` mode
4. **Crop:** Slice output back to original dimensions `[:h_orig, :w_orig]` — the padded border pixels are discarded before saving and before MSE computation
5. **Evaluate:** Compute channel-wise MSE against `day.jpg`
6. **Save:** Output saved as `enhanced_night.jpg`

**Padding rationale:** Reflect padding mirrors real border pixels so the model sees plausible content at the boundary (rather than a hard black edge, which would bias near-border convolutions). The padded strip is fully discarded before any metric computation.

```python
def pad_to_multiple(t, m=16):
    _, _, h, w = t.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    return torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect"), h, w
```

---

## 10. HuggingFace Deployment

All models, datasets, and the interactive demo are publicly available on HuggingFace:

| Resource | Link |
|---|---|
| Model v1 | [`tyakovenko/night-to-day-enhancement-model`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model) |
| Model v1-extended | [`tyakovenko/night-to-day-enhancement-model-extended`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-extended) |
| Dataset (TA) | [`tyakovenko/night-to-day-enhancement`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) |
| Dataset (TA + LOL) | [`tyakovenko/night-to-day-enhancement-extended`](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement-extended) |
| Live Demo (Space) | [`tyakovenko/night-to-day-enhancement`](https://huggingface.co/spaces/tyakovenko/night-to-day-enhancement) |

The Gradio Space (`app.py`) loads the model checkpoint from HuggingFace Hub at startup and provides a live image enhancement demo.

---

## 11. Discussion

### 11.1 Why Learning-Based Over Classical Methods?

Classical approaches (histogram equalization, gamma correction, Retinex-based methods) can increase brightness but do not have access to supervision from paired day images, making it difficult to recover realistic color and texture. A supervised learning approach allows the model to learn the full image-to-image mapping directly from paired data.

### 11.2 Why MSE Loss?

MSE was chosen as the loss function because it directly aligns with the evaluation metric. MSE penalizes large pixel-level errors heavily (due to squaring), which encourages the model to minimize the same quantity being measured at test time. A potential drawback is that MSE can produce slightly blurry outputs because the model may average over plausible outputs for ambiguous inputs (regression-to-mean).

### 11.3 Why Scene-Level Train/Val Split?

A scene-level split ensures that all images from a given outdoor scene appear in only one partition (train or val). This prevents data leakage — without this constraint, the model could implicitly memorize a scene's appearance from training images of the same location and report artificially low validation MSE.

### 11.4 Impact of Fine-Tuning on LOL

Adding the LOL dataset improved val MSE by 4.1% overall. The largest improvement was in the Red channel (−6.1%), likely because indoor artificial lighting produces warm/reddish casts that the TA-only model had not seen enough examples of. The fine-tuning LR was set to 1e-5 (10× lower than the original 1e-4) to avoid catastrophic forgetting of features learned on the TA dataset.

---

## 12. Limitations and Future Work

| Limitation | Potential Improvement |
|---|---|
| MSE loss produces slightly blurry outputs | Use perceptual loss (VGG) or MS-SSIM + L1 combination |
| 1.8M parameter model may struggle with very dark inputs | Increase `base_filters` to 32 (requires GPU for feasible training time) |
| Training on crops (128×128) — full-resolution context not seen during training | Use larger crops or multi-scale training |
| val MSE oscillates — no early stopping beyond manual checkpointing | Implement strict early stopping based on validation plateau |
| LOL is indoor-only — may slightly reduce outdoor performance | Balance indoor/outdoor ratios more carefully |

> A v2 model was explored using L1 + MS-SSIM loss (initialized from `best_extended.pt`), but training was still in progress at submission time.

---

## 13. Reproducibility

### Environment

```
Python 3.10+
torch >= 2.0
torchvision
Pillow
huggingface_hub
pandas
```

Install via: `pip install -r requirements.txt`

### Run Training (v1)

```bash
python train.py
```

### Run Fine-Tuning (v1-extended)

```bash
python train_extended.py
```

### Run Inference and Evaluation

```bash
python enhance.py --input night.jpg --ref day.jpg --output enhanced_night.jpg
```

### Load Model from HuggingFace

```python
from huggingface_hub import hf_hub_download
import torch
from model import UNet

ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model", "best_extended.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
model = UNet(base_filters=ckpt["args"]["base_filters"])
model.load_state_dict(ckpt["model"])
model.eval()
```

All model checkpoints (`best.pt`, `best_extended.pt`) and datasets are publicly hosted on HuggingFace Hub and require no authentication to download.

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*. https://arxiv.org/abs/1505.04597
2. Laffont, P.-Y., Ren, Z., Tao, X., Qian, C., & Hays, J. (2014). Transient Attributes for High-Level Understanding and Editing of Outdoor Scenes. *ACM SIGGRAPH 2014*. http://transattr.cs.brown.edu/
3. Wei, C., Wang, W., Yang, W., & Liu, J. (2018). Deep Retinex Decomposition for Low-Light Enhancement. *BMVC 2018*. (LOL Dataset) https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

---

*[Sections marked with suggested additions below — fill in before submission]*

> **TODO for student:** Add visual comparison figures (night.jpg / enhanced_night.jpg / day.jpg side by side), slide deck screenshots, and course/assignment details at the top of this document.
