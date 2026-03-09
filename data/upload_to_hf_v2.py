#!/usr/bin/env python3
"""
Upload curated dataset to Hugging Face Hub using upload_folder (batched commits).
This avoids the per-file commit rate limit.
"""

import os
import sys
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, create_repo

BASE_DIR    = Path("/home/taya/night-to-day-imgConversion")
IMAGES_ROOT = BASE_DIR / "transientAttributesDataset/imageAlignedLD"
MANIFEST    = BASE_DIR / "low_light_manifest.csv"
REPORT      = BASE_DIR / "data-analyst-report.md"
STAGING_DIR = BASE_DIR / "hf_staging"
REPO_ID     = "tyakovenko/night-to-day-enhancement"

api = HfApi()

# Verify authentication
try:
    whoami = api.whoami()
    print(f"Authenticated as: {whoami.get('name', 'unknown')}")
except Exception as e:
    print(f"ERROR: Not authenticated — {e}")
    sys.exit(1)

valid_df = pd.read_csv(MANIFEST)
print(f"Manifest rows: {len(valid_df)}")
print(f"Unique low-light images: {valid_df['low_light_image'].nunique()}")
print(f"Unique day-target images: {valid_df['day_target_image'].nunique()}")

# Build staging directory
print(f"\nBuilding staging directory: {STAGING_DIR}")
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
STAGING_DIR.mkdir(parents=True)

# Copy manifest and report
shutil.copy(MANIFEST, STAGING_DIR / "low_light_manifest.csv")
shutil.copy(REPORT, STAGING_DIR / "data-analyst-report.md")
print("Copied manifest and report")

# Copy low-light images
ll_dir = STAGING_DIR / "low_light"
missing_ll = 0
copied_ll = 0

for _, row in valid_df.iterrows():
    src = IMAGES_ROOT / row["low_light_image"]
    dst = ll_dir / row["low_light_image"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, dst)
        copied_ll += 1
    else:
        print(f"  MISSING low-light: {src}")
        missing_ll += 1

print(f"Low-light images staged: {copied_ll} (missing: {missing_ll})")

# Copy day-target images (unique only)
day_dir = STAGING_DIR / "day"
missing_day = 0
copied_day = 0
seen_day = set()

for _, row in valid_df.iterrows():
    if row["day_target_image"] in seen_day:
        continue
    src = IMAGES_ROOT / row["day_target_image"]
    dst = day_dir / row["day_target_image"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy(src, dst)
        copied_day += 1
        seen_day.add(row["day_target_image"])
    else:
        print(f"  MISSING day-target: {src}")
        missing_day += 1

print(f"Day-target images staged: {copied_day} (missing: {missing_day})")

# Count staged files
staged_count = sum(1 for _ in STAGING_DIR.rglob("*") if _.is_file())
staged_size_mb = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file()) / (1024*1024)
print(f"\nTotal staged files: {staged_count} ({staged_size_mb:.1f} MB)")

# Create/access repo
print(f"\nCreating/accessing repo: {REPO_ID}")
try:
    create_repo(repo_id=REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    print("  Repo ready")
except Exception as e:
    print(f"  create_repo info: {e}")

# Upload entire folder in one batched operation
print(f"\nUploading folder to Hub (batched commit)...")
try:
    result = api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add curated low-light/day pairs from Transient Attributes Dataset",
        ignore_patterns=["*.pyc", "__pycache__"],
    )
    print(f"Upload successful!")
    print(f"Commit URL: {result}")
    hf_url = f"https://huggingface.co/datasets/{REPO_ID}"
    print(f"Dataset URL: {hf_url}")
    upload_success = True
except Exception as e:
    print(f"Upload failed: {type(e).__name__}: {e}")
    upload_success = False
    hf_url = None

# Write status
status_path = BASE_DIR / "hf_upload_status.txt"
with open(status_path, "w") as f:
    if upload_success:
        f.write(f"status: SUCCESS\n")
        f.write(f"url: {hf_url}\n")
        f.write(f"low_light_uploaded: {copied_ll}\n")
        f.write(f"day_target_uploaded: {copied_day}\n")
        f.write(f"total_staged_mb: {staged_size_mb:.1f}\n")
    else:
        f.write(f"status: FAILED\n")
        f.write(f"low_light_staged: {copied_ll}\n")
        f.write(f"day_staged: {copied_day}\n")

print(f"\nStatus written to: {status_path}")
print(f"Upload {'SUCCEEDED' if upload_success else 'FAILED'}")
