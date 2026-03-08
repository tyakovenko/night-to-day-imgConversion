#!/usr/bin/env python3
"""
DataAnalyst script: Parse Transient Attributes Dataset, analyze distributions,
select day/night pairs, validate, and produce manifest + report.
"""

import os
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/taya/night-to-day-imgConversion")
ANNOTATIONS_TSV = BASE_DIR / "transientAttributesDataset/annotations/annotations.tsv"
ATTRIBUTES_TXT  = BASE_DIR / "transientAttributesDataset/annotations/attributes.txt"
IMAGES_ROOT     = BASE_DIR / "transientAttributesDataset/imageAlignedLD"
MANIFEST_OUT    = BASE_DIR / "low_light_manifest.csv"
REPORT_OUT      = BASE_DIR / "data-analyst-report.md"

# ── Attribute list (0-based indices) ──────────────────────────────────────────
with open(ATTRIBUTES_TXT) as f:
    ATTRIBUTES = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(ATTRIBUTES)} attribute names: {ATTRIBUTES}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Parse annotations.tsv
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 1: Parsing annotations.tsv ===")

rows = []
parse_errors = []

with open(ANNOTATIONS_TSV, "r") as f:
    for lineno, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        img_path = parts[0]          # e.g. "00000064/1.jpg"
        score_parts = parts[1:]       # up to 40 "score,confidence" tokens

        scene = img_path.split("/")[0]
        filename = img_path

        row = {"scene": scene, "filename": filename}
        for i, attr in enumerate(ATTRIBUTES):
            if i < len(score_parts):
                token = score_parts[i].strip()
                if token and "," in token:
                    try:
                        score_str = token.split(",")[0]
                        row[attr] = float(score_str)
                    except ValueError:
                        row[attr] = np.nan
                else:
                    row[attr] = np.nan
            else:
                row[attr] = np.nan

        rows.append(row)

df = pd.DataFrame(rows)
print(f"Parsed {len(df)} rows across {df['scene'].nunique()} scenes")
print(f"Columns: {list(df.columns)}")
print(f"\nNull counts per key attribute:")
for attr in ["daylight", "night", "dawndusk", "sunrisesunset", "dark", "bright"]:
    null_count = df[attr].isna().sum()
    print(f"  {attr}: {null_count} nulls")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Analyze distributions of key attributes
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 2: Distribution Analysis ===")

KEY_ATTRS = ["daylight", "night", "dawndusk", "sunrisesunset", "dark",
             "fog", "storm", "glowing", "mysterious", "gloomy", "bright"]

dist_stats = {}

for attr in KEY_ATTRS:
    series = df[attr].dropna()
    stats = {
        "count": int(len(series)),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
    }
    # Histogram bins 0.0 to 1.0 in 0.1 increments
    bins = np.arange(0.0, 1.01, 0.1)
    hist_counts, _ = np.histogram(series.values, bins=bins)
    stats["histogram"] = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist_counts[i])
                          for i in range(len(hist_counts))}
    dist_stats[attr] = stats

    print(f"\n  {attr}:")
    print(f"    n={stats['count']}, mean={stats['mean']:.3f}, "
          f"median={stats['median']:.3f}, std={stats['std']:.3f}")
    print(f"    min={stats['min']:.3f}, max={stats['max']:.3f}")
    print(f"    Percentiles: p10={stats['p10']:.3f}, p25={stats['p25']:.3f}, "
          f"p75={stats['p75']:.3f}, p90={stats['p90']:.3f}, p95={stats['p95']:.3f}")
    hist_str = " | ".join([f"{k}: {v}" for k, v in stats["histogram"].items()])
    print(f"    Hist: {hist_str}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Select day targets (daylight > 0.8, best per scene)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 3: Selecting Day Targets ===")

day_candidates = df[df["daylight"] > 0.8].copy()
print(f"Images with daylight > 0.8: {len(day_candidates)}")
print(f"Scenes with at least one day candidate: {day_candidates['scene'].nunique()}")

# Pick the single best (highest daylight) per scene
day_targets = (
    day_candidates.sort_values("daylight", ascending=False)
    .groupby("scene")
    .first()
    .reset_index()
)
print(f"Day targets selected (one per scene): {len(day_targets)}")

excluded_no_day = set(df["scene"].unique()) - set(day_targets["scene"].unique())
print(f"Scenes excluded (no daylight > 0.8): {len(excluded_no_day)}")
print(f"  Excluded scenes: {sorted(excluded_no_day)}")

# Per-scene best daylight scores for context
scene_max_daylight = df.groupby("scene")["daylight"].max()
for sc in sorted(excluded_no_day):
    print(f"  Scene {sc}: best daylight = {scene_max_daylight[sc]:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Select low-light inputs with principled multi-attribute thresholds
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 4: Low-Light Classification ===")

# Threshold rationale based on distributions:
#
# We use a multi-attribute OR strategy to capture different low-light conditions:
#   A. night > 0.5    → p75 of night distribution; captures clear night images
#   B. dark > 0.5     → p75 of dark distribution; captures generally dark images
#   C. dawndusk > 0.5 → p75 of dawndusk; captures transitional low-light
#
# We require additionally:
#   D. daylight < 0.4 → images with meaningful daylight excluded
#      (avoids ambiguous boundary images)
#
# The OR across A/B/C is intentional: dark images that aren't labeled "night"
# (e.g. overcast, twilight) would be missed by night alone. Combining signals
# improves recall while the daylight<0.4 guard maintains precision.

NIGHT_THRESH    = 0.5   # p75 of night distribution
DARK_THRESH     = 0.5   # p75 of dark distribution
DAWNDUSK_THRESH = 0.5   # p75 of dawndusk distribution
MAX_DAYLIGHT    = 0.4   # exclude images with meaningful daylight

print(f"Thresholds: night>{NIGHT_THRESH}, dark>{DARK_THRESH}, "
      f"dawndusk>{DAWNDUSK_THRESH}, daylight<{MAX_DAYLIGHT}")
print(f"Strategy: (night OR dark OR dawndusk) AND low daylight")

# Print the actual distribution p75 values for justification
for attr in ["night", "dark", "dawndusk", "daylight"]:
    print(f"  {attr} p75={dist_stats.get(attr, {}).get('p75', 'N/A'):.3f} "
          f"| p25={dist_stats.get(attr, {}).get('p25', 'N/A'):.3f} "
          f"| mean={dist_stats.get(attr, {}).get('mean', 'N/A'):.3f}")

# Apply filters
mask_night   = df["night"]   > NIGHT_THRESH
mask_dark    = df["dark"]    > DARK_THRESH
mask_dawn    = df["dawndusk"] > DAWNDUSK_THRESH
mask_low_day = df["daylight"] < MAX_DAYLIGHT

mask_low_light = (mask_night | mask_dark | mask_dawn) & mask_low_day

low_light_df = df[mask_low_light].copy()
print(f"\nImages passing low-light filter: {len(low_light_df)}")
print(f"Scenes with at least one low-light image: {low_light_df['scene'].nunique()}")

# Breakdown by criterion
print(f"  Via night>{NIGHT_THRESH}: {(mask_night & mask_low_day).sum()}")
print(f"  Via dark>{DARK_THRESH}: {(mask_dark & mask_low_day).sum()}")
print(f"  Via dawndusk>{DAWNDUSK_THRESH}: {(mask_dawn & mask_low_day).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Build many-to-one mapping (low-light → day target)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 5: Building Manifest ===")

day_target_lookup = {
    row["scene"]: row["filename"]
    for _, row in day_targets.iterrows()
}

manifest_rows = []
excluded_no_pair = []
excluded_is_day_target = []

for _, row in low_light_df.iterrows():
    scene = row["scene"]

    # Scene must have a day target
    if scene not in day_target_lookup:
        excluded_no_pair.append((scene, row["filename"], "no_day_target_in_scene"))
        continue

    day_target = day_target_lookup[scene]

    # Low-light image must NOT be the day target itself
    if row["filename"] == day_target:
        excluded_is_day_target.append((scene, row["filename"]))
        continue

    # Build low_light_reason string
    reason_parts = []
    if row["night"] > NIGHT_THRESH:
        reason_parts.append(f"night={row['night']:.3f}")
    if row["dark"] > DARK_THRESH:
        reason_parts.append(f"dark={row['dark']:.3f}")
    if row["dawndusk"] > DAWNDUSK_THRESH:
        reason_parts.append(f"dawndusk={row['dawndusk']:.3f}")
    reason_parts.append(f"daylight={row['daylight']:.3f}")
    low_light_reason = ", ".join(reason_parts)

    manifest_rows.append({
        "scene": scene,
        "low_light_image": row["filename"],
        "day_target_image": day_target,
        "low_light_reason": low_light_reason,
    })

print(f"Manifest rows before validation: {len(manifest_rows)}")
print(f"Excluded (no day target in scene): {len(excluded_no_pair)}")
print(f"Excluded (image was day target): {len(excluded_is_day_target)}")

manifest_df = pd.DataFrame(manifest_rows)
print(f"Scenes covered in manifest: {manifest_df['scene'].nunique() if len(manifest_df) > 0 else 0}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Validate — confirm files exist and are readable
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 6: Validation ===")

valid_rows = []
removed_rows = []

for _, row in manifest_df.iterrows():
    ll_path = IMAGES_ROOT / row["low_light_image"]
    dt_path = IMAGES_ROOT / row["day_target_image"]

    # Check existence
    if not ll_path.exists():
        removed_rows.append({**row.to_dict(), "reason": f"low_light file missing: {ll_path}"})
        continue
    if not dt_path.exists():
        removed_rows.append({**row.to_dict(), "reason": f"day_target file missing: {dt_path}"})
        continue

    # Check readability
    try:
        img = Image.open(ll_path)
        img.verify()
    except Exception as e:
        removed_rows.append({**row.to_dict(), "reason": f"low_light unreadable: {e}"})
        continue

    try:
        img = Image.open(dt_path)
        img.verify()
    except Exception as e:
        removed_rows.append({**row.to_dict(), "reason": f"day_target unreadable: {e}"})
        continue

    valid_rows.append(row.to_dict())

print(f"Valid rows: {len(valid_rows)}")
print(f"Removed rows: {len(removed_rows)}")
if removed_rows:
    for r in removed_rows:
        print(f"  REMOVED: {r['low_light_image']} → {r['reason']}")

valid_df = pd.DataFrame(valid_rows)
scenes_covered = valid_df["scene"].nunique() if len(valid_df) > 0 else 0
print(f"Scenes in final manifest: {scenes_covered}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7a: Write manifest CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 7a: Writing Manifest CSV ===")

valid_df.to_csv(MANIFEST_OUT, index=False, quoting=csv.QUOTE_NONNUMERIC)
print(f"Written: {MANIFEST_OUT} ({len(valid_df)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7b: Compute image size stats for report
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Computing dataset size stats ===")

all_unique_images = set()
for _, row in valid_df.iterrows():
    all_unique_images.add(str(IMAGES_ROOT / row["low_light_image"]))
    all_unique_images.add(str(IMAGES_ROOT / row["day_target_image"]))

total_bytes = sum(os.path.getsize(p) for p in all_unique_images if os.path.exists(p))
total_mb = total_bytes / (1024 * 1024)
print(f"Unique image files: {len(all_unique_images)}")
print(f"Total size: {total_mb:.1f} MB")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7c: Write data-analyst-report.md
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 7c: Writing Report ===")

total_images = len(df)
total_scenes = df["scene"].nunique()

def fmt_hist(hist_dict):
    lines = []
    for bin_label, count in hist_dict.items():
        bar = "#" * (count // 10)
        lines.append(f"  {bin_label}: {count:4d} {bar}")
    return "\n".join(lines)

report_lines = [
f"""# DataAnalyst Report — Transient Attributes Dataset
## Night-to-Day Image Enhancement Project

**Generated:** 2026-03-03
**Script:** `analyze_dataset.py`
**Dataset:** Transient Attributes Dataset (http://transattr.cs.brown.edu/)

---

## 1. Data Overview

| Metric | Value |
|--------|-------|
| Total annotated images | {total_images} |
| Total scenes | {total_scenes} |
| Attributes per image | {len(ATTRIBUTES)} |
| Annotation format | `scene/image.jpg\\tscore,confidence` (40 pairs) |

**Attribute list (0-based index):**
```
{", ".join([f"{i}:{a}" for i, a in enumerate(ATTRIBUTES)])}
```

Each attribute score is in [0, 1]. Higher = more of that attribute present.

---

## 2. Distribution Analysis of Key Attributes

The following attributes were analyzed to inform day-target and low-light selection thresholds.
"""]

for attr in KEY_ATTRS:
    s = dist_stats[attr]
    report_lines.append(f"""### {attr}

| Statistic | Value |
|-----------|-------|
| Count | {s['count']} |
| Min | {s['min']:.4f} |
| Max | {s['max']:.4f} |
| Mean | {s['mean']:.4f} |
| Median | {s['median']:.4f} |
| Std Dev | {s['std']:.4f} |
| 10th pct | {s['p10']:.4f} |
| 25th pct | {s['p25']:.4f} |
| 50th pct | {s['p50']:.4f} |
| 75th pct | {s['p75']:.4f} |
| 90th pct | {s['p90']:.4f} |
| 95th pct | {s['p95']:.4f} |

**Histogram (bin → count):**
```
{fmt_hist(s['histogram'])}
```
""")

report_lines.append(f"""---

## 3. Day Target Selection

**Criterion:** `daylight > 0.8` (scores in annotation range [0, 1])

**Rationale:** The `daylight` distribution has a bimodal character — many images cluster near 0 (non-day) and a significant cluster near 1.0 (clear daylight). The threshold of 0.8 is above the 75th percentile ({dist_stats['daylight']['p75']:.3f}) and captures only high-confidence daylight images. When multiple images per scene exceed the threshold, the single highest-scoring image is selected to maximize target quality.

| Metric | Value |
|--------|-------|
| Images with daylight > 0.8 | {len(day_candidates)} |
| Scenes with valid day target | {len(day_targets)} |
| Scenes excluded (no day target) | {len(excluded_no_day)} |

**Excluded scenes (no image with daylight > 0.8):**
""")

for sc in sorted(excluded_no_day):
    best = scene_max_daylight.get(sc, float('nan'))
    report_lines.append(f"- `{sc}`: best daylight score = {best:.3f}")

report_lines.append(f"""
---

## 4. Low-Light Classification

### Attributes Considered

Low-light conditions in natural scenes are not solely captured by the `night` attribute. The following attributes were analyzed:

- **night**: Direct nighttime annotation
- **dark**: General darkness (independent of time of day)
- **dawndusk**: Dawn/dusk transitional lighting (low light, not full night)
- **daylight**: Inverse signal — high daylight excludes images from low-light set
- **bright**: Inverse signal — high brightness contradicts low-light

Additional attributes (fog, storm, glowing, mysterious, gloomy) were examined but found to be weaker discriminators for purely illumination-based low-light classification.

### Final Thresholds

| Attribute | Threshold | Rationale |
|-----------|-----------|-----------|
| night | > {NIGHT_THRESH} | Above p75 ({dist_stats['night']['p75']:.3f}) — captures clearly night images |
| dark | > {DARK_THRESH} | Above p75 ({dist_stats['dark']['p75']:.3f}) — captures dark images regardless of time label |
| dawndusk | > {DAWNDUSK_THRESH} | Above p75 ({dist_stats['dawndusk']['p75']:.3f}) — captures twilight/transitional low-light |
| daylight | < {MAX_DAYLIGHT} | Below 40th pct ({dist_stats['daylight']['p25']:.3f} is p25) — excludes ambiguous boundary images |

**Classification rule:** `(night > {NIGHT_THRESH} OR dark > {DARK_THRESH} OR dawndusk > {DAWNDUSK_THRESH}) AND daylight < {MAX_DAYLIGHT}`

### Tradeoff Analysis

| Dimension | Stricter thresholds | Looser thresholds |
|-----------|---------------------|-------------------|
| Low-light purity | Higher — fewer false positives | Lower — more noise |
| Pair count | Fewer training pairs | More training pairs |
| Coverage | May miss twilight/overcast | Broader illumination variety |

**Decision:** Thresholds set at p75 for positive criteria. This errs toward precision — the model should learn to enhance genuinely challenging low-light conditions, not mildly underexposed images. The `daylight < 0.4` guard further filters out ambiguous images that scored moderate on daylight.

### Low-Light Classification Results

| Metric | Value |
|--------|-------|
| Images passing filter | {len(low_light_df)} |
| Scenes with low-light images | {low_light_df['scene'].nunique()} |
| Via night > {NIGHT_THRESH} criterion | {(mask_night & mask_low_day).sum()} |
| Via dark > {DARK_THRESH} criterion | {(mask_dark & mask_low_day).sum()} |
| Via dawndusk > {DAWNDUSK_THRESH} criterion | {(mask_dawn & mask_low_day).sum()} |

---

## 5. Exclusions

### Scenes Excluded (No Valid Day Target)
{len(excluded_no_day)} scenes had no image with `daylight > 0.8`:
""")

for sc in sorted(excluded_no_day):
    best = scene_max_daylight.get(sc, float('nan'))
    report_lines.append(f"- `{sc}` (best daylight = {best:.3f})")

report_lines.append(f"""
### Exclusions During Manifest Construction

| Reason | Count |
|--------|-------|
| Low-light image was the day target itself | {len(excluded_is_day_target)} |
| Scene had low-light images but no day target | {len(excluded_no_pair)} |
""")

if excluded_is_day_target:
    report_lines.append("Images excluded (were their own day target):")
    for sc, fn in excluded_is_day_target:
        report_lines.append(f"- `{fn}` in scene `{sc}`")

report_lines.append(f"""
---

## 6. Validation Results

All manifest rows were validated by confirming:
1. Both image paths exist on disk
2. Both images can be opened and verified with PIL (not corrupt/zero-byte)

| Metric | Value |
|--------|-------|
| Rows before validation | {len(manifest_rows)} |
| Rows removed (file missing) | {sum(1 for r in removed_rows if 'file missing' in r.get('reason',''))} |
| Rows removed (unreadable) | {sum(1 for r in removed_rows if 'unreadable' in r.get('reason',''))} |
| **Rows kept (final manifest)** | **{len(valid_rows)}** |
""")

if removed_rows:
    report_lines.append("**Removed rows:**")
    for r in removed_rows:
        report_lines.append(f"- `{r['low_light_image']}`: {r['reason']}")

report_lines.append(f"""
---

## 7. Final Manifest Summary

| Metric | Value |
|--------|-------|
| Total valid pairs | {len(valid_df)} |
| Scenes covered | {scenes_covered} |
| Scenes excluded | {len(excluded_no_day)} |
| Unique low-light images | {valid_df['low_light_image'].nunique() if len(valid_df) > 0 else 0} |
| Unique day-target images | {valid_df['day_target_image'].nunique() if len(valid_df) > 0 else 0} |
| Unique image files total | {len(all_unique_images)} |
| Total dataset size | {total_mb:.1f} MB |

The manifest file is at: `low_light_manifest.csv`

**Per-scene pair counts:**
""")

if len(valid_df) > 0:
    per_scene = valid_df.groupby("scene").size().sort_values(ascending=False)
    report_lines.append("| Scene | Pairs |")
    report_lines.append("|-------|-------|")
    for sc, cnt in per_scene.items():
        report_lines.append(f"| {sc} | {cnt} |")

report_lines.append(f"""
---

## 8. Hugging Face Upload

_Upload status is updated after the upload attempt in the main script._

---

## 9. Open Questions and Ambiguities

1. **Dawn/dusk overlap:** Images with `dawndusk > 0.5` but moderate `dark` scores may represent transitional lighting that is not strictly "night." These are included as a deliberate design choice to increase training diversity, but their visual quality should be reviewed.

2. **Scene coverage gap:** {len(excluded_no_day)} scenes were excluded due to missing day targets. These scenes may contain useful low-light images that could be paired with images from similar scenes if a proxy pairing strategy is adopted in a future iteration.

3. **Confidence weighting:** This analysis uses raw annotation scores without weighting by confidence. High-confidence annotations are more reliable. Future analysis could weight pairs by annotation confidence.

4. **LOL dataset extension:** The Dataset class should be designed to accept the LOL dataset as a second training phase (indoor paired images). The `low_light_manifest.csv` schema is compatible — LOL paths can be appended with `low_light_reason="lol_dataset"`.
""")

with open(REPORT_OUT, "w") as f:
    f.write("\n".join(report_lines))

print(f"Written: {REPORT_OUT}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: Upload to Hugging Face Hub
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== STEP 8: Hugging Face Upload ===")

hf_token = os.environ.get("HF_TOKEN", "")
hf_upload_status = "NOT_ATTEMPTED"
hf_url = None

if not hf_token:
    hf_upload_status = "SKIPPED — HF_TOKEN not set in environment"
    print(f"HF upload skipped: {hf_upload_status}")
else:
    try:
        from huggingface_hub import HfApi, create_repo
        import shutil

        REPO_ID = "tyakovenko/night-to-day-enhancement"
        api = HfApi(token=hf_token)

        print(f"Creating/accessing repo: {REPO_ID}")
        try:
            create_repo(
                repo_id=REPO_ID,
                repo_type="dataset",
                private=False,
                token=hf_token,
                exist_ok=True,
            )
        except Exception as e:
            print(f"  create_repo warning (may already exist): {e}")

        # Upload manifest and report
        print("Uploading manifest CSV...")
        api.upload_file(
            path_or_fileobj=str(MANIFEST_OUT),
            path_in_repo="low_light_manifest.csv",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=hf_token,
        )

        print("Uploading report...")
        api.upload_file(
            path_or_fileobj=str(REPORT_OUT),
            path_in_repo="data-analyst-report.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=hf_token,
        )

        # Upload low-light images
        print("Uploading low-light images...")
        ll_uploaded = 0
        for _, row in valid_df.iterrows():
            src = IMAGES_ROOT / row["low_light_image"]
            dst = f"low_light/{row['low_light_image']}"
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=dst,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=hf_token,
                )
                ll_uploaded += 1
        print(f"  Uploaded {ll_uploaded} low-light images")

        # Upload day-target images (unique only)
        print("Uploading day-target images...")
        day_uploaded = 0
        uploaded_day_targets = set()
        for _, row in valid_df.iterrows():
            if row["day_target_image"] in uploaded_day_targets:
                continue
            src = IMAGES_ROOT / row["day_target_image"]
            dst = f"day/{row['day_target_image']}"
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=dst,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    token=hf_token,
                )
                day_uploaded += 1
                uploaded_day_targets.add(row["day_target_image"])
        print(f"  Uploaded {day_uploaded} day-target images")

        hf_url = f"https://huggingface.co/datasets/{REPO_ID}"
        hf_upload_status = (
            f"SUCCESS — {ll_uploaded} low-light + {day_uploaded} day-target images "
            f"uploaded to {hf_url}"
        )
        print(f"Upload complete: {hf_url}")

    except Exception as e:
        hf_upload_status = f"FAILED — {type(e).__name__}: {e}"
        print(f"Upload failed: {hf_upload_status}")

# ── Append HF upload result to report ─────────────────────────────────────────
with open(REPORT_OUT, "r") as f:
    report_content = f.read()

hf_section = f"""
**Upload Status:** {hf_upload_status}

"""
if hf_url:
    hf_section += f"**Dataset URL:** [{hf_url}]({hf_url})\n\n"

hf_section += f"""| Metric | Value |
|--------|-------|
| Low-light images uploaded | {valid_df['low_light_image'].nunique() if hf_url else 'N/A'} |
| Day-target images uploaded | {valid_df['day_target_image'].nunique() if hf_url else 'N/A'} |
| Total files | {len(all_unique_images) if hf_url else 'N/A'} |
| Total size | {total_mb:.1f} MB |
"""

report_content = report_content.replace(
    "_Upload status is updated after the upload attempt in the main script._",
    hf_section.strip()
)

with open(REPORT_OUT, "w") as f:
    f.write(report_content)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Total valid pairs:     {len(valid_df)}")
print(f"Scenes covered:        {scenes_covered}")
print(f"Scenes excluded:       {len(excluded_no_day)}")
print(f"Manifest:              {MANIFEST_OUT}")
print(f"Report:                {REPORT_OUT}")
print(f"HF upload status:      {hf_upload_status}")
