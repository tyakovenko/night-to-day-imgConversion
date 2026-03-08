"""
Dataset class for low-light → day image pairs.

Reads pairs from low_light_manifest.csv.
Images are fetched from HF Hub (cached locally on first access).

Column schema:
    scene             - scene folder ID
    low_light_image   - relative path, e.g. "00000064/3.jpg"
    day_target_image  - relative path, e.g. "00000064/12.jpg"
    low_light_reason  - which thresholds triggered classification
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

HF_REPO_ID = "tyakovenko/night-to-day-enhancement"          # original TA repo (default)
HF_REPO_TYPE = "dataset"
MANIFEST_PATH = Path(__file__).parent / "low_light_manifest.csv"
EXTENDED_MANIFEST_PATH = Path(__file__).parent / "extended_manifest.csv"


def _load_image_rgb(path: str) -> np.ndarray:
    """Load JPEG/PNG as float32 RGB array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _fetch_image(hf_prefix: str, rel_path: str, repo_id: str = HF_REPO_ID) -> str:
    """
    Download image from HF Hub (cached to ~/.cache/huggingface/hub).
    Returns local cache path.

    hf_prefix: "low_light" or "day"
    rel_path:  e.g. "00000064/3.jpg" or "lol/1.png"
    repo_id:   HF dataset repo to fetch from (defaults to original TA repo)
    """
    repo_path = f"{hf_prefix}/{rel_path}"
    return hf_hub_download(
        repo_id=repo_id,
        filename=repo_path,
        repo_type=HF_REPO_TYPE,
    )


class LowLightDataset(Dataset):
    """
    Paired (low-light, day-target) dataset.

    Args:
        manifest_df: DataFrame with columns from low_light_manifest.csv.
        crop_size:   Random crop size for training. None = full resolution.
        augment:     If True, apply horizontal flip augmentation.
        cache_dir:   Optional local directory to cache downloaded images.
                     Defaults to HF hub cache (~/.cache/huggingface/hub).
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        crop_size: Optional[int] = 256,
        augment: bool = True,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Fetch from HF Hub (returns cached local path after first download).
        # repo_id column allows rows from different HF repos (e.g. LOL vs TA).
        repo_id = row.get("repo_id", HF_REPO_ID)
        ll_path = _fetch_image("low_light", row["low_light_image"], repo_id)
        dt_path = _fetch_image("day",       row["day_target_image"], repo_id)

        # Load as float32 RGB in [0, 1]
        ll_img = _load_image_rgb(ll_path)
        dt_img = _load_image_rgb(dt_path)

        # Random crop (training only)
        if self.crop_size is not None:
            ll_img, dt_img = self._random_crop(ll_img, dt_img, self.crop_size)

        # Horizontal flip
        if self.augment and random.random() > 0.5:
            ll_img = np.fliplr(ll_img).copy()
            dt_img = np.fliplr(dt_img).copy()

        # HWC -> CHW tensor
        ll_t = torch.from_numpy(ll_img.transpose(2, 0, 1))
        dt_t = torch.from_numpy(dt_img.transpose(2, 0, 1))

        return ll_t, dt_t

    @staticmethod
    def _random_crop(ll: np.ndarray, dt: np.ndarray, size: int):
        h, w = ll.shape[:2]
        if h < size or w < size:
            # Pad if image is smaller than crop size
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            ll = np.pad(ll, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            dt = np.pad(dt, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = ll.shape[:2]
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return ll[y:y + size, x:x + size], dt[y:y + size, x:x + size]


def build_splits(
    manifest_path: Path = MANIFEST_PATH,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scene-level train/val split.

    Splitting at scene level (not image level) prevents leakage: all images
    from a given scene appear in only one split.
    """
    df = pd.read_csv(manifest_path)
    scenes = sorted(df["scene"].unique())
    rng = random.Random(seed)
    rng.shuffle(scenes)

    n_val = max(1, int(len(scenes) * val_fraction))
    val_scenes = set(scenes[:n_val])
    train_scenes = set(scenes[n_val:])

    train_df = df[df["scene"].isin(train_scenes)].reset_index(drop=True)
    val_df = df[df["scene"].isin(val_scenes)].reset_index(drop=True)

    return train_df, val_df
