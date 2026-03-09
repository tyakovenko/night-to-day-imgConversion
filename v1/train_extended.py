"""
Fine-tuning script — U-Net low-light enhancement (extended dataset).

Starts from checkpoints/best.pt (trained on Transient Attributes) and
fine-tunes on the combined TA + LOL dataset (extended_manifest.csv).

Usage:
    python train_extended.py [--epochs 20] [--batch-size 8] [--crop-size 128]
                             [--lr 1e-5] [--workers 4]
                             [--init-checkpoint checkpoints/best.pt]
                             [--checkpoint-dir checkpoints/]
                             [--resume checkpoints/last_extended.pt]

Outputs:
    - checkpoints/best_extended.pt   (best val MSE)
    - checkpoints/last_extended.pt   (latest epoch, for resume)
    - experiment_log_extended.csv    (per-epoch metrics)
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import (
    LowLightDataset,
    build_splits,
    EXTENDED_MANIFEST_PATH,
    HF_REPO_ID,
    _fetch_image,
)
from model import UNet, count_parameters


# ── Evaluation ────────────────────────────────────────────────────────────────

def channel_mse(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Channel-wise MSE in float64, values in [0, 1]."""
    p = pred.detach().cpu().numpy().astype(np.float64)
    t = target.detach().cpu().numpy().astype(np.float64)
    mse_r = float(np.mean((p[:, 0] - t[:, 0]) ** 2))
    mse_g = float(np.mean((p[:, 1] - t[:, 1]) ** 2))
    mse_b = float(np.mean((p[:, 2] - t[:, 2]) ** 2))
    return {"R": mse_r, "G": mse_g, "B": mse_b, "avg": (mse_r + mse_g + mse_b) / 3.0}


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for ll, dt in loader:
            ll, dt = ll.to(device), dt.to(device)
            pred = model(ll)
            loss = criterion(pred, dt)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * ll.size(0)
            all_preds.append(pred.detach().cpu())
            all_targets.append(dt.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    cmse = channel_mse(torch.cat(all_preds), torch.cat(all_targets))
    return avg_loss, cmse


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune U-Net on combined TA + LOL dataset"
    )
    parser.add_argument("--epochs",           type=int,   default=20)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--crop-size",        type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=1e-5,
                        help="Learning rate (default 1e-5, 10x lower than original training)")
    parser.add_argument("--workers",          type=int,   default=4)
    parser.add_argument("--val-fraction",     type=float, default=0.15)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--checkpoint-dir",   type=str,   default="checkpoints")
    parser.add_argument("--init-checkpoint",  type=str,   default="checkpoints/best.pt",
                        help="Pretrained checkpoint to fine-tune from")
    parser.add_argument("--resume",           type=str,   default=None,
                        help="Resume extended training from a last_extended.pt checkpoint")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Low-Light Enhancement — Extended Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Device:           {device}")
    print(f"  Init checkpoint:  {args.init_checkpoint}")
    print(f"  Manifest:         {EXTENDED_MANIFEST_PATH}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Crop size:        {args.crop_size}×{args.crop_size}")
    print(f"  LR:               {args.lr}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    if not EXTENDED_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Extended manifest not found: {EXTENDED_MANIFEST_PATH}\n"
            "Run the LOL upload + manifest generation step first."
        )

    print("Building train/val splits (scene-level) from extended manifest ...")
    train_df, val_df = build_splits(
        manifest_path=EXTENDED_MANIFEST_PATH,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    n_ta_train  = (train_df["low_light_reason"] != "lol_dataset").sum()
    n_lol_train = (train_df["low_light_reason"] == "lol_dataset").sum()
    n_ta_val    = (val_df["low_light_reason"]   != "lol_dataset").sum()
    n_lol_val   = (val_df["low_light_reason"]   == "lol_dataset").sum()
    print(f"  Train: {len(train_df):>4} pairs  (TA={n_ta_train}, LOL={n_lol_train})")
    print(f"  Val:   {len(val_df):>4} pairs  (TA={n_ta_val},  LOL={n_lol_val})\n")

    # ── Prefetch all images to local HF cache (parallel, repo-aware) ─────────
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    # Build (prefix, rel_path, repo_id) tuples — deduplicated
    seen = set()
    tasks = []
    for _, row in all_df.iterrows():
        repo_id = row.get("repo_id", HF_REPO_ID)
        for prefix, col in [("low_light", "low_light_image"), ("day", "day_target_image")]:
            key = (prefix, row[col], repo_id)
            if key not in seen:
                seen.add(key)
                tasks.append(key)

    total = len(tasks)
    print(f"Pre-fetching {total} unique images from HF Hub (cached after first run) ...")
    t_fetch = time.time()
    done = 0

    def _fetch(task):
        prefix, rel, repo_id = task
        _fetch_image(prefix, rel, repo_id)
        return rel

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch, t): t for t in tasks}
        for f in as_completed(futures):
            done += 1
            if done % 100 == 0 or done == total:
                elapsed_f = time.time() - t_fetch
                rate = done / elapsed_f
                remaining = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total}  ({rate:.0f} img/s  ETA {remaining:.0f}s)")
    print(f"  Done in {time.time()-t_fetch:.0f}s\n")

    train_ds = LowLightDataset(train_df, crop_size=args.crop_size, augment=True)
    val_ds   = LowLightDataset(val_df,   crop_size=args.crop_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    # ── Model — load pretrained weights ──────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch  = 1
    best_val_mse = float("inf")

    if args.resume and Path(args.resume).exists():
        # Resume interrupted extended training
        ckpt = torch.load(args.resume, map_location=device)
        base_filters = ckpt.get("args", {}).get("base_filters", 16)
        model = UNet(base_filters=base_filters).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"Resumed extended training from {args.resume}  (epoch {ckpt['epoch']})\n")
    else:
        # Fine-tune from best.pt
        if not Path(args.init_checkpoint).exists():
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_checkpoint}")
        init_ckpt    = torch.load(args.init_checkpoint, map_location=device)
        base_filters = init_ckpt.get("args", {}).get("base_filters", 16)
        model = UNet(base_filters=base_filters).to(device)
        model.load_state_dict(init_ckpt["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Initialized from {args.init_checkpoint}  "
              f"(epoch {init_ckpt.get('epoch','?')}, "
              f"val MSE {init_ckpt.get('val_mse', 0):.6f})")
        print(f"Model parameters: {count_parameters(model):,}\n")

    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-7
    )

    # ── Experiment log ───────────────────────────────────────────────────────
    log_path   = Path("experiment_log_extended.csv")
    log_fields = [
        "epoch", "train_loss", "train_mse_R", "train_mse_G", "train_mse_B", "train_mse_avg",
        "val_loss", "val_mse_R", "val_mse_G", "val_mse_B", "val_mse_avg",
        "lr", "epoch_time_s",
    ]
    write_header = (not log_path.exists()) or (start_epoch == 1)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if write_header:
        log_writer.writeheader()

    # ── Column header ────────────────────────────────────────────────────────
    W = 110
    print("─" * W)
    print(f"{'Ep':>4} │ {'TrainLoss':>10} {'Tr_MSE':>8} │"
          f" {'ValLoss':>9} {'Va_MSE':>8} │"
          f" {'MSE_R':>8} {'MSE_G':>8} {'MSE_B':>8} │"
          f" {'LR':>8} {'Time':>6}  {'ETA':>8}")
    print("─" * W)

    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, tr_cmse = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )
        val_loss, val_cmse = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False
        )

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        eta_s   = np.mean(epoch_times) * (args.epochs - epoch)
        eta_str = f"{eta_s/3600:.1f}h" if eta_s >= 3600 else f"{eta_s/60:.0f}m"
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_cmse["avg"])

        print(
            f"{epoch:>4} │"
            f" {train_loss:>10.6f} {tr_cmse['avg']:>8.6f} │"
            f" {val_loss:>9.6f} {val_cmse['avg']:>8.6f} │"
            f" {val_cmse['R']:>8.6f} {val_cmse['G']:>8.6f} {val_cmse['B']:>8.6f} │"
            f" {current_lr:>8.1e} {elapsed:>5.0f}s  {eta_str:>8}"
        )

        # ── Save checkpoints (extended names — originals untouched) ──────────
        ckpt_state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_mse":      val_cmse["avg"],
            "best_val_mse": best_val_mse,
            "args":         {**vars(args), "base_filters": base_filters},
        }
        torch.save(ckpt_state, ckpt_dir / "last_extended.pt")

        if val_cmse["avg"] < best_val_mse:
            best_val_mse = val_cmse["avg"]
            ckpt_state["best_val_mse"] = best_val_mse
            torch.save(ckpt_state, ckpt_dir / "best_extended.pt")
            print(f"       ↳ new best  val MSE: {best_val_mse:.6f}"
                  f"  → saved to {ckpt_dir}/best_extended.pt")

        # ── Log ──────────────────────────────────────────────────────────────
        log_writer.writerow({
            "epoch":         epoch,
            "train_loss":    f"{train_loss:.6f}",
            "train_mse_R":   f"{tr_cmse['R']:.6f}",
            "train_mse_G":   f"{tr_cmse['G']:.6f}",
            "train_mse_B":   f"{tr_cmse['B']:.6f}",
            "train_mse_avg": f"{tr_cmse['avg']:.6f}",
            "val_loss":      f"{val_loss:.6f}",
            "val_mse_R":     f"{val_cmse['R']:.6f}",
            "val_mse_G":     f"{val_cmse['G']:.6f}",
            "val_mse_B":     f"{val_cmse['B']:.6f}",
            "val_mse_avg":   f"{val_cmse['avg']:.6f}",
            "lr":            f"{current_lr:.2e}",
            "epoch_time_s":  f"{elapsed:.1f}",
        })
        log_file.flush()

    log_file.close()
    print("─" * W)
    print(f"\nFine-tuning complete.")
    print(f"  Best val MSE avg:  {best_val_mse:.6f}")
    print(f"  Best checkpoint:   {ckpt_dir}/best_extended.pt")
    print(f"  Last checkpoint:   {ckpt_dir}/last_extended.pt")
    print(f"  Experiment log:    {log_path}\n")


if __name__ == "__main__":
    main()
