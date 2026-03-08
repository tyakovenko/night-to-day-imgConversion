"""
Training script for U-Net low-light enhancement.

Usage:
    python train.py [--epochs 30] [--batch-size 4] [--crop-size 256]
                    [--base-filters 32] [--lr 1e-4] [--workers 4]
                    [--checkpoint-dir checkpoints/] [--resume checkpoints/last.pt]

Outputs:
    - Per-epoch loss table to stdout
    - checkpoints/best.pt  (best val MSE)
    - checkpoints/last.pt  (latest epoch, for resume)
    - experiment_log.csv   (full per-epoch metrics)
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader

from dataset import LowLightDataset, build_splits, _fetch_image
from model import UNet, count_parameters


# ── Evaluation ────────────────────────────────────────────────────────────────

def channel_mse(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Channel-wise MSE in float64, values in [0, 1].
    Returns dict with keys: R, G, B, avg.
    """
    p = pred.detach().cpu().numpy().astype(np.float64)   # (N, C, H, W)
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
    preds_cat  = torch.cat(all_preds,   dim=0)
    tgts_cat   = torch.cat(all_targets, dim=0)
    cmse = channel_mse(preds_cat, tgts_cat)
    return avg_loss, cmse


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for low-light enhancement")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--crop-size",      type=int,   default=256)
    parser.add_argument("--base-filters",   type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--workers",        type=int,   default=4)
    parser.add_argument("--val-fraction",   type=float, default=0.15)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--resume",         type=str,   default=None)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Low-Light Enhancement — U-Net Training")
    print(f"{'='*60}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Crop size:    {args.crop_size}×{args.crop_size}")
    print(f"  Base filters: {args.base_filters}")
    print(f"  LR:           {args.lr}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Building train/val splits (scene-level) ...")
    train_df, val_df = build_splits(val_fraction=args.val_fraction, seed=args.seed)
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"  Train: {len(train_df):>4} pairs  |  {train_df['scene'].nunique()} scenes")
    print(f"  Val:   {len(val_df):>4} pairs  |  {val_df['scene'].nunique()} scenes\n")

    # ── Prefetch all images to local HF cache (parallel) ─────────────────────
    unique_ll  = all_df["low_light_image"].unique().tolist()
    unique_day = all_df["day_target_image"].unique().tolist()
    tasks = [("low_light", p) for p in unique_ll] + [("day", p) for p in unique_day]
    total = len(tasks)
    print(f"Pre-fetching {total} unique images from HF Hub (cached after first run) ...")
    t_fetch = time.time()
    done = 0

    def _fetch(args_tuple):
        prefix, rel = args_tuple
        _fetch_image(prefix, rel)
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

    # ── Model ────────────────────────────────────────────────────────────────
    model = UNet(base_filters=args.base_filters).to(device)
    print(f"Model parameters: {count_parameters(model):,}\n")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    start_epoch  = 1
    best_val_mse = float("inf")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']})\n")

    # ── Experiment log ───────────────────────────────────────────────────────
    log_path   = Path("experiment_log.csv")
    log_fields = [
        "epoch", "train_loss", "train_mse_R", "train_mse_G", "train_mse_B", "train_mse_avg",
        "val_loss",   "val_mse_R",   "val_mse_G",   "val_mse_B",   "val_mse_avg",
        "lr", "epoch_time_s",
    ]
    write_header = (not log_path.exists()) or (start_epoch == 1)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if write_header:
        log_writer.writeheader()

    # ── Warm-up benchmark ────────────────────────────────────────────────────
    print("Fetching first batch (triggers HF Hub downloads on first run) ...")
    t_bench = time.time()
    for ll, dt in train_loader:
        ll, dt = ll.to(device), dt.to(device)
        with torch.no_grad():
            _ = model(ll)
        break
    bench_s = time.time() - t_bench
    est_epoch_s = bench_s * (len(train_loader) + len(val_loader))
    print(f"  First batch:            {bench_s:.1f}s")
    print(f"  Est. time per epoch:    ~{est_epoch_s/60:.1f} min")
    print(f"  Est. total ({args.epochs} epochs): ~{est_epoch_s * args.epochs / 3600:.1f} h\n")

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

        # ── Save checkpoints ─────────────────────────────────────────────────
        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_mse":      val_cmse["avg"],
            "best_val_mse": best_val_mse,
            "args":         vars(args),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_cmse["avg"] < best_val_mse:
            best_val_mse = val_cmse["avg"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"       ↳ new best  val MSE: {best_val_mse:.6f}  → saved to {ckpt_dir}/best.pt")

        # ── Log ──────────────────────────────────────────────────────────────
        log_writer.writerow({
            "epoch":           epoch,
            "train_loss":      f"{train_loss:.6f}",
            "train_mse_R":     f"{tr_cmse['R']:.6f}",
            "train_mse_G":     f"{tr_cmse['G']:.6f}",
            "train_mse_B":     f"{tr_cmse['B']:.6f}",
            "train_mse_avg":   f"{tr_cmse['avg']:.6f}",
            "val_loss":        f"{val_loss:.6f}",
            "val_mse_R":       f"{val_cmse['R']:.6f}",
            "val_mse_G":       f"{val_cmse['G']:.6f}",
            "val_mse_B":       f"{val_cmse['B']:.6f}",
            "val_mse_avg":     f"{val_cmse['avg']:.6f}",
            "lr":              f"{current_lr:.2e}",
            "epoch_time_s":    f"{elapsed:.1f}",
        })
        log_file.flush()

    log_file.close()
    print("─" * W)
    print(f"\nTraining complete.")
    print(f"  Best val MSE avg:  {best_val_mse:.6f}")
    print(f"  Best checkpoint:   {ckpt_dir}/best.pt")
    print(f"  Last checkpoint:   {ckpt_dir}/last.pt")
    print(f"  Experiment log:    {log_path}\n")


if __name__ == "__main__":
    main()
