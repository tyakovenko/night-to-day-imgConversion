#!/usr/bin/env python3
"""
Upload curated dataset to Hugging Face Hub at tyakovenko/night-to-day-enhancement.
Reads low_light_manifest.csv to determine which images to upload.
"""

import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR   = Path("/home/taya/night-to-day-imgConversion")
IMAGES_ROOT = BASE_DIR / "transientAttributesDataset/imageAlignedLD"
MANIFEST   = BASE_DIR / "low_light_manifest.csv"
REPORT     = BASE_DIR / "data-analyst-report.md"
REPO_ID    = "tyakovenko/night-to-day-enhancement"

from huggingface_hub import HfApi, create_repo

hf_token = os.environ.get("HF_TOKEN", "")

# Fall back to cached token if env var is not set
api = HfApi(token=hf_token if hf_token else None)
try:
    whoami = api.whoami()
    print(f"Authenticated as: {whoami.get('name', 'unknown')}")
except Exception as e:
    print(f"ERROR: Not authenticated — {e}")
    sys.exit(1)
valid_df = pd.read_csv(MANIFEST)
print(f"Manifest rows: {len(valid_df)}")

# Create repo
print(f"Creating/accessing repo: {REPO_ID}")
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    print("  Repo ready")
except Exception as e:
    print(f"  create_repo: {e}")

# Upload manifest
print("Uploading low_light_manifest.csv ...")
api.upload_file(
    path_or_fileobj=str(MANIFEST),
    path_in_repo="low_light_manifest.csv",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  Done")

# Upload report
print("Uploading data-analyst-report.md ...")
api.upload_file(
    path_or_fileobj=str(REPORT),
    path_in_repo="data-analyst-report.md",
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("  Done")

# Upload low-light images in batches to avoid too many individual API calls
print("Uploading low-light images ...")
ll_uploaded = 0
ll_errors = 0
for _, row in valid_df.iterrows():
    src = IMAGES_ROOT / row["low_light_image"]
    dst = f"low_light/{row['low_light_image']}"
    if not src.exists():
        print(f"  MISSING: {src}")
        ll_errors += 1
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        ll_uploaded += 1
        if ll_uploaded % 100 == 0:
            print(f"  ... {ll_uploaded} low-light images uploaded")
    except Exception as e:
        print(f"  ERROR uploading {src}: {e}")
        ll_errors += 1

print(f"Low-light images: {ll_uploaded} uploaded, {ll_errors} errors")

# Upload day-target images (unique only)
print("Uploading day-target images ...")
day_uploaded = 0
day_errors = 0
uploaded_day_targets = set()

for _, row in valid_df.iterrows():
    if row["day_target_image"] in uploaded_day_targets:
        continue
    src = IMAGES_ROOT / row["day_target_image"]
    dst = f"day/{row['day_target_image']}"
    if not src.exists():
        print(f"  MISSING: {src}")
        day_errors += 1
        continue
    try:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        day_uploaded += 1
        uploaded_day_targets.add(row["day_target_image"])
        if day_uploaded % 20 == 0:
            print(f"  ... {day_uploaded} day-target images uploaded")
    except Exception as e:
        print(f"  ERROR uploading {src}: {e}")
        day_errors += 1

print(f"Day-target images: {day_uploaded} uploaded, {day_errors} errors")

hf_url = f"https://huggingface.co/datasets/{REPO_ID}"
print(f"\nUpload complete!")
print(f"Dataset URL: {hf_url}")
print(f"Total: {ll_uploaded} low-light + {day_uploaded} day-target images")

# Write upload result to a status file
with open(BASE_DIR / "hf_upload_status.txt", "w") as f:
    f.write(f"status: SUCCESS\n")
    f.write(f"url: {hf_url}\n")
    f.write(f"low_light_uploaded: {ll_uploaded}\n")
    f.write(f"day_target_uploaded: {day_uploaded}\n")
    f.write(f"ll_errors: {ll_errors}\n")
    f.write(f"day_errors: {day_errors}\n")

print("Status written to hf_upload_status.txt")
