#!/usr/bin/env python3
"""
Upload remaining files to HF Hub using upload_large_folder which handles
rate limiting and retries automatically.
"""

import os
import sys
from pathlib import Path

BASE_DIR    = Path("/home/taya/night-to-day-imgConversion")
STAGING_DIR = BASE_DIR / "hf_staging"
REPO_ID     = "tyakovenko/night-to-day-enhancement"

from huggingface_hub import HfApi

api = HfApi()

# Verify authentication
try:
    whoami = api.whoami()
    print(f"Authenticated as: {whoami.get('name', 'unknown')}")
except Exception as e:
    print(f"ERROR: Not authenticated — {e}")
    sys.exit(1)

if not STAGING_DIR.exists():
    print(f"ERROR: Staging dir not found: {STAGING_DIR}")
    sys.exit(1)

staged_count = sum(1 for f in STAGING_DIR.rglob("*") if f.is_file())
staged_mb = sum(f.stat().st_size for f in STAGING_DIR.rglob("*") if f.is_file()) / (1024*1024)
print(f"Staged files: {staged_count} ({staged_mb:.1f} MB)")
print(f"Target repo: {REPO_ID}")

# Check what's already uploaded
try:
    existing = list(api.list_repo_files(REPO_ID, repo_type="dataset"))
    print(f"Already on Hub: {len(existing)} files")
except Exception as e:
    print(f"Could not list existing files: {e}")
    existing = []

print(f"\nStarting upload_large_folder (handles rate limits automatically)...")
try:
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(STAGING_DIR),
        num_workers=2,
        print_report=True,
    )
    hf_url = f"https://huggingface.co/datasets/{REPO_ID}"
    print(f"\nUpload complete!")
    print(f"Dataset URL: {hf_url}")

    status_path = BASE_DIR / "hf_upload_status.txt"
    with open(status_path, "w") as f:
        f.write(f"status: SUCCESS\n")
        f.write(f"url: {hf_url}\n")
        f.write(f"files_uploaded: {staged_count}\n")
        f.write(f"total_mb: {staged_mb:.1f}\n")
    print(f"Status written to: {status_path}")

except Exception as e:
    print(f"Upload failed: {type(e).__name__}: {e}")
    status_path = BASE_DIR / "hf_upload_status.txt"
    with open(status_path, "w") as f:
        f.write(f"status: FAILED\n")
        f.write(f"error: {type(e).__name__}: {e}\n")
        f.write(f"staged_files: {staged_count}\n")
