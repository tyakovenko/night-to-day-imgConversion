"""
Training script v4 — Lamp-suppressing losses + Global Context + Color warmup.

Changes vs train_v3.py:
  - Loss:    V4Loss (WeightedL1 + LogL1 + MS-SSIM + ColorLoss with staged warmup)
             Both WeightedL1 and LogL1 down-weight bright lamp pixels, fixing the
             streetlamp-amplification problem from v3.
  - Arch:    UNet(residual=True, use_global_context=True) optional — adds a
             GlobalContextEncoder that injects full-image brightness stats into
             the bottleneck so the model can distinguish lamp pixels from sunlight.
  - Aug:     Color jitter (gamma Uniform(0.6, 1.4) on low-light input only) prevents
             overfitting to the TA sensor brightness profile.
  - LR:      CosineAnnealingWarmRestarts(T_0=10) — smoother convergence.
  - Init:    Warm-starts from best.pt (cleaner MSE baseline) not best_extended.pt.
  - Saves to checkpoints/best_v4.pt / last_v4.pt
  - Logs to experiment_log_v4.csv

Usage:
    python train_v4.py [--epochs 30] [--batch-size 4] [--crop-size 176]
                       [--lr 1e-4] [--color-weight 0.5] [--color-warmup-epochs 5]
                       [--w-weighted-l1 0.16] [--w-log-l1 0.16] [--w-ssim 0.68]
                       [--global-context] [--augment-color]
                       [--workers 4]
                       [--init-checkpoint checkpoints/best.pt]
                       [--resume checkpoints/last_v4.pt]
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import (
    LowLightDataset,
    build_splits,
    EXTENDED_MANIFEST_PATH,
    MANIFEST_PATH,
    HF_REPO_ID,
    _fetch_image,
)
from model import UNet, count_parameters
from losses import V4Loss


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

def run_epoch(model, loader, criterion, optimizer, device,
              is_train: bool, use_global_context: bool):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if use_global_context:
                ll, dt, stats = batch
                stats = stats.to(device)
            else:
                ll, dt = batch
                stats = None

            ll, dt = ll.to(device), dt.to(device)

            # model forward — pass global stats if available
            pred = model(ll, global_stats=stats)

            # criterion needs the original input for WeightedL1
            loss = criterion(pred, dt, ll)

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
        description="Train U-Net v4 with lamp-suppressing losses + global context"
    )
    parser.add_argument("--epochs",               type=int,   default=30)
    parser.add_argument("--batch-size",           type=int,   default=4)
    parser.add_argument("--crop-size",            type=int,   default=176,
                        help="176 is MS-SSIM minimum; larger = more context")
    parser.add_argument("--lr",                   type=float, default=1e-4)
    # Loss weights
    parser.add_argument("--w-weighted-l1",        type=float, default=0.16,
                        help="Weight for input-luminance-weighted L1 term")
    parser.add_argument("--w-log-l1",             type=float, default=0.16,
                        help="Weight for log-domain L1 term")
    parser.add_argument("--w-ssim",               type=float, default=0.68,
                        help="Weight for (1 - MS-SSIM) term")
    parser.add_argument("--color-weight",         type=float, default=0.5,
                        help="Final weight for YCbCr ColorLoss")
    parser.add_argument("--color-warmup-epochs",  type=int,   default=5,
                        help="ColorLoss is 0 for first N epochs, then linear ramp "
                             "over the next N epochs to full weight")
    # Architecture
    parser.add_argument("--global-context",       action="store_true",
                        help="Add GlobalContextEncoder to bottleneck (scene-level stats)")
    # Augmentation
    parser.add_argument("--augment-color",        action="store_true",
                        help="Apply gamma jitter to low-light input (Uniform(0.6, 1.4))")
    # Training infra
    parser.add_argument("--workers",              type=int,   default=4)
    parser.add_argument("--val-fraction",         type=float, default=0.15)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--checkpoint-dir",       type=str,   default="checkpoints")
    parser.add_argument("--init-checkpoint",      type=str,   default=None,
                        help="Checkpoint to warm-start from. "
                             "Defaults to best.pt. strict=False so new GlobalContextEncoder "
                             "weights initialise randomly while encoder/decoder transfer.")
    parser.add_argument("--resume",               type=str,   default=None,
                        help="Resume v4 training from last_v4.pt")
    parser.add_argument("--manifest",             type=str,   default=None,
                        help="Path to manifest CSV. Defaults to extended_manifest.csv "
                             "if it exists, else low_light_manifest.csv")
    args = parser.parse_args()

    # Resolve manifest
    if args.manifest is not None:
        manifest_path = Path(args.manifest)
    elif EXTENDED_MANIFEST_PATH.exists():
        manifest_path = EXTENDED_MANIFEST_PATH
    else:
        manifest_path = MANIFEST_PATH

    # Resolve default init checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    if args.init_checkpoint is None:
        args.init_checkpoint = str(ckpt_dir / "best.pt")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  Low-Light Enhancement — v4 Training")
    print(f"  (WeightedL1 + LogL1 + MS-SSIM + ColorLoss w/ staged warmup)")
    print(f"{'='*70}")
    print(f"  Device:               {device}")
    print(f"  Init checkpoint:      {args.init_checkpoint}")
    print(f"  Manifest:             {manifest_path}")
    print(f"  Epochs:               {args.epochs}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Crop size:            {args.crop_size}×{args.crop_size}")
    print(f"  LR:                   {args.lr}  (CosineAnnealingWarmRestarts T_0=10)")
    print(f"  Loss weights:         wl1={args.w_weighted_l1}  log_l1={args.w_log_l1}"
          f"  ssim={args.w_ssim}")
    print(f"  Color weight:         {args.color_weight}"
          f"  (warmup {args.color_warmup_epochs} epochs, ramp {args.color_warmup_epochs} epochs)")
    print(f"  Global context:       {args.global_context}")
    print(f"  Color augmentation:   {args.augment_color}")
    print(f"{'='*70}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print("Building train/val splits (scene-level) ...")
    train_df, val_df = build_splits(
        manifest_path=manifest_path,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    n_ta_train  = (train_df["low_light_reason"] != "lol_dataset").sum()
    n_lol_train = (train_df["low_light_reason"] == "lol_dataset").sum()
    n_ta_val    = (val_df["low_light_reason"]   != "lol_dataset").sum()
    n_lol_val   = (val_df["low_light_reason"]   == "lol_dataset").sum()
    print(f"  Train: {len(train_df):>4} pairs  (TA={n_ta_train}, LOL={n_lol_train})")
    print(f"  Val:   {len(val_df):>4} pairs  (TA={n_ta_val},  LOL={n_lol_val})\n")

    # ── Prefetch images to HF cache ───────────────────────────────────────────
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    seen, tasks = set(), []
    for _, row in all_df.iterrows():
        repo_id = row.get("repo_id", HF_REPO_ID)
        for prefix, col in [("low_light", "low_light_image"), ("day", "day_target_image")]:
            key = (prefix, row[col], repo_id)
            if key not in seen:
                seen.add(key)
                tasks.append(key)

    total = len(tasks)
    print(f"Pre-fetching {total} unique images from HF Hub ...")
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

    train_ds = LowLightDataset(
        train_df,
        crop_size=args.crop_size,
        augment=True,
        augment_color=args.augment_color,
        return_global_stats=args.global_context,
    )
    val_ds = LowLightDataset(
        val_df,
        crop_size=args.crop_size,
        augment=False,
        augment_color=False,
        return_global_stats=args.global_context,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch  = 1
    best_val_mse = float("inf")

    if args.resume and Path(args.resume).exists():
        # Resume a v4 checkpoint
        ckpt = torch.load(args.resume, map_location=device)
        saved_args = ckpt.get("args", {})
        base_filters = saved_args.get("base_filters", 16)
        use_gc = saved_args.get("global_context", args.global_context)
        model = UNet(
            base_filters=base_filters, residual=True,
            use_global_context=use_gc,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        print(f"Resumed v4 training from {args.resume}  (epoch {ckpt['epoch']})\n")
    else:
        # Warm-start from best.pt (sigmoid, non-residual).
        # strict=False: missing GlobalContextEncoder keys initialise randomly;
        # output activation mismatch (sigmoid vs tanh) is parameter-free, so
        # the weight transfer is clean.
        if not Path(args.init_checkpoint).exists():
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_checkpoint}")
        init_ckpt    = torch.load(args.init_checkpoint, map_location=device)
        base_filters = init_ckpt.get("args", {}).get("base_filters", 16)
        model = UNet(
            base_filters=base_filters, residual=True,
            use_global_context=args.global_context,
        ).to(device)
        model.load_state_dict(init_ckpt["model"], strict=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Initialized (residual=True, global_context={args.global_context}) "
              f"from {args.init_checkpoint}  "
              f"(epoch {init_ckpt.get('epoch','?')}, "
              f"val MSE {init_ckpt.get('val_mse', 0):.6f})")
        print(f"Model parameters: {count_parameters(model):,}\n")

    criterion = V4Loss(
        w_weighted_l1=args.w_weighted_l1,
        w_log_l1=args.w_log_l1,
        w_ssim=args.w_ssim,
        color_weight=args.color_weight,
    ).to(device)

    # CosineAnnealingWarmRestarts: LR follows a cosine curve, restarting every
    # T_0 epochs. Avoids plateau-based LR reduction getting stuck early.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-7
    )

    # ── Experiment log ────────────────────────────────────────────────────────
    log_path   = Path("experiment_log_v4.csv")
    log_fields = [
        "epoch", "color_scale",
        "train_loss", "train_mse_R", "train_mse_G", "train_mse_B", "train_mse_avg",
        "val_loss", "val_mse_R", "val_mse_G", "val_mse_B", "val_mse_avg",
        "lr", "epoch_time_s",
    ]
    write_header = (not log_path.exists()) or (start_epoch == 1)
    log_file   = open(log_path, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    if write_header:
        log_writer.writeheader()

    # ── Column header ─────────────────────────────────────────────────────────
    W = 120
    print("─" * W)
    print(f"{'Ep':>4} │ {'CScale':>6} │ {'TrainLoss':>10} {'Tr_MSE':>8} │"
          f" {'ValLoss':>9} {'Va_MSE':>8} │"
          f" {'MSE_R':>8} {'MSE_G':>8} {'MSE_B':>8} │"
          f" {'LR':>8} {'Time':>6}  {'ETA':>8}")
    print("─" * W)

    epoch_times = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Staged ColorLoss: 0 for first color_warmup_epochs, then linear ramp
        # over the next color_warmup_epochs until it reaches 1.0.
        # Example with warmup=5: ep1-5 → 0, ep6→0.2, ep7→0.4, ..., ep10→1.0
        wu = args.color_warmup_epochs
        color_scale = max(0.0, min(1.0, (epoch - wu) / max(1, wu)))
        criterion.set_color_scale(color_scale)

        train_loss, tr_cmse = run_epoch(
            model, train_loader, criterion, optimizer, device,
            is_train=True, use_global_context=args.global_context,
        )
        val_loss, val_cmse = run_epoch(
            model, val_loader, criterion, optimizer, device,
            is_train=False, use_global_context=args.global_context,
        )

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        eta_s   = np.mean(epoch_times) * (args.epochs - epoch)
        eta_str = f"{eta_s/3600:.1f}h" if eta_s >= 3600 else f"{eta_s/60:.0f}m"
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(epoch)  # CosineAnnealingWarmRestarts uses epoch index

        print(
            f"{epoch:>4} │"
            f" {color_scale:>6.2f} │"
            f" {train_loss:>10.6f} {tr_cmse['avg']:>8.6f} │"
            f" {val_loss:>9.6f} {val_cmse['avg']:>8.6f} │"
            f" {val_cmse['R']:>8.6f} {val_cmse['G']:>8.6f} {val_cmse['B']:>8.6f} │"
            f" {current_lr:>8.1e} {elapsed:>5.0f}s  {eta_str:>8}"
        )

        # ── Save checkpoints (v4 names) ───────────────────────────────────────
        ckpt_state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "val_mse":      val_cmse["avg"],
            "best_val_mse": best_val_mse,
            "args": {
                **vars(args),
                "base_filters":     base_filters,
                "global_context":   args.global_context,
            },
        }
        torch.save(ckpt_state, ckpt_dir / "last_v4.pt")

        if val_cmse["avg"] < best_val_mse:
            best_val_mse = val_cmse["avg"]
            ckpt_state["best_val_mse"] = best_val_mse
            torch.save(ckpt_state, ckpt_dir / "best_v4.pt")
            print(f"       ↳ new best  val MSE: {best_val_mse:.6f}"
                  f"  → saved to {ckpt_dir}/best_v4.pt")

        # ── Log ───────────────────────────────────────────────────────────────
        log_writer.writerow({
            "epoch":         epoch,
            "color_scale":   f"{color_scale:.2f}",
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
    print(f"\nv4 training complete.")
    print(f"  Best val MSE avg:  {best_val_mse:.6f}")
    print(f"  Best checkpoint:   {ckpt_dir}/best_v4.pt")
    print(f"  Last checkpoint:   {ckpt_dir}/last_v4.pt")
    print(f"  Experiment log:    {log_path}\n")
    print("Next steps:")
    print("  python enhance.py --input night.jpg --reference day.jpg "
          "--checkpoint checkpoints/best_v4.pt")
    print("  # Compare streetlamp pixels vs enhanced_night_v3.jpg")


if __name__ == "__main__":
    main()
