"""
Inference script for U-Net low-light enhancement.

Usage:
    # Enhance only (no reference)
    python enhance.py --input night.png --output enhanced.png --checkpoint checkpoints/best.pt

    # Enhance + evaluate against ground truth
    python enhance.py --input night.png --reference day.png --output enhanced.png \
                      --checkpoint checkpoints/best.pt

Output: enhanced image saved to --output.
        If --reference provided: prints channel-wise MSE + avg MSE.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import UNet


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as float32 RGB in [0, 1]. Preserves original size."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def image_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """HWC float32 -> 1CHW tensor."""
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """1CHW tensor -> HWC uint8."""
    arr = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).round().astype(np.uint8)


def channel_mse_eval(pred: np.ndarray, ref: np.ndarray) -> dict:
    """
    Channel-wise MSE in float64, values in [0, 1].
    pred/ref: HWC float32 arrays.
    Returns dict with keys: R, G, B, avg.
    """
    assert pred.shape == ref.shape, (
        f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}"
    )
    p = pred.astype(np.float64)
    r = ref.astype(np.float64)
    mse_r = float(np.mean((p[:, :, 0] - r[:, :, 0]) ** 2))
    mse_g = float(np.mean((p[:, :, 1] - r[:, :, 1]) ** 2))
    mse_b = float(np.mean((p[:, :, 2] - r[:, :, 2]) ** 2))
    mse_avg = (mse_r + mse_g + mse_b) / 3.0
    return {"R": mse_r, "G": mse_g, "B": mse_b, "avg": mse_avg}


def pad_to_multiple(t: torch.Tensor, multiple: int = 16):
    """
    Pad a 1CHW tensor so H and W are divisible by `multiple`.
    Uses reflect padding so border pixels mirror real image content,
    minimising edge distortion in the padded region.
    Returns (padded_tensor, (pad_h, pad_w)) so the caller can crop back.
    """
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        # Pad bottom and right only — easy to crop by [:h_orig, :w_orig]
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, (pad_h, pad_w)


def enhance(input_path: str, checkpoint_path: str, base_filters: int | None = None) -> np.ndarray:
    """
    Run inference on a single image at full resolution.
    Returns enhanced image as HWC float32 array in [0, 1].

    base_filters is auto-detected from checkpoint's saved args when not specified.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Auto-detect base_filters from checkpoint to avoid flag/checkpoint mismatch
    if base_filters is None:
        base_filters = ckpt.get("args", {}).get("base_filters", 16)

    model = UNet(base_filters=base_filters).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    input_arr = load_image_rgb(input_path)
    h_orig, w_orig = input_arr.shape[:2]
    input_t = image_to_tensor(input_arr).to(device)

    # U-Net has 4 MaxPool layers → H and W must be divisible by 16.
    # Eval images (1024×737) have H%16=1, so 15px of reflect padding is added
    # to the bottom; the padded strip is cropped back after inference and never
    # appears in the saved output or MSE computation.
    input_t, (pad_h, pad_w) = pad_to_multiple(input_t, multiple=16)

    with torch.no_grad():
        output_t = model(input_t)

    # Crop back to original dimensions before returning
    if pad_h > 0 or pad_w > 0:
        output_t = output_t[:, :, :h_orig, :w_orig]

    output_arr = output_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(output_arr, 0.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Enhance low-light image with U-Net")
    parser.add_argument("--input",       required=True,  help="Input low-light image path")
    parser.add_argument("--output",      default=None,   help="Output path (default: enhanced_<input>)")
    parser.add_argument("--reference",   default=None,   help="Day reference image for MSE evaluation")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--base-filters", type=int, default=None,
                        help="Base filters for U-Net (auto-detected from checkpoint if omitted)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Default output path
    if args.output is None:
        inp = Path(args.input)
        args.output = str(inp.parent / f"enhanced_{inp.name}")

    print(f"Input:      {args.input}")
    print(f"Checkpoint: {args.checkpoint}")

    enhanced = enhance(args.input, args.checkpoint, args.base_filters)

    # Save output — preserve input dimensions (no resize)
    out_uint8 = (np.clip(enhanced, 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(out_uint8).save(args.output)
    print(f"Output:     {args.output}  ({enhanced.shape[1]}×{enhanced.shape[0]})")

    # Evaluation
    if args.reference:
        ref = load_image_rgb(args.reference)
        assert enhanced.shape == ref.shape, (
            f"Dimension mismatch: enhanced {enhanced.shape} vs reference {ref.shape}\n"
            "Output must match reference dimensions exactly."
        )
        mse = channel_mse_eval(enhanced, ref)
        print(f"\n{'='*40}")
        print(f"  Channel-wise MSE (float64, [0,1] scale)")
        print(f"{'='*40}")
        print(f"  MSE_R:   {mse['R']:.6f}")
        print(f"  MSE_G:   {mse['G']:.6f}")
        print(f"  MSE_B:   {mse['B']:.6f}")
        print(f"  MSE_avg: {mse['avg']:.6f}")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
