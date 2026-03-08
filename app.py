"""
Gradio UI — Low-Light to Day Image Enhancement
Hosted at: huggingface.co/spaces/tyakovenko/night-to-day-enhancement

Model: U-Net (base_filters=16), trained on Transient Attributes Dataset.
Checkpoint: tyakovenko/night-to-day-enhancement-model / best.pt
"""

import numpy as np
from PIL import Image
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from skimage.metrics import structural_similarity as ssim_fn
from model import UNet

MODEL_REPO = "tyakovenko/night-to-day-enhancement-model"

# Lazy model state — loaded on first inference request, not at import time.
# This prevents a DNS/network failure during Space container init from
# crashing the app before the UI even starts.
_model = None
_model_loaded = False
_model_attempted = False


def get_model():
    """Load model on first call; return cached instance on subsequent calls."""
    global _model, _model_loaded, _model_attempted
    if _model_attempted:
        return _model, _model_loaded
    _model_attempted = True
    try:
        ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename="best.pt", repo_type="model")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_filters = ckpt.get("args", {}).get("base_filters", 16)
        model = UNet(base_filters=base_filters)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Model loaded from {MODEL_REPO} "
              f"(epoch {ckpt.get('epoch', '?')}, val MSE {ckpt.get('val_mse', 0):.4f})")
        _model, _model_loaded = model, True
    except Exception as e:
        print(f"Model load failed: {e}")
        _model, _model_loaded = None, False
    return _model, _model_loaded


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """
    Compute all evaluation metrics between enhanced image and reference.

    Args:
        pred: HWC uint8 enhanced image
        ref:  HWC uint8 reference (day) image — must match pred shape

    Returns dict with keys: mse, mse_r, mse_g, mse_b, ssim, fid_note
    """
    if pred.shape != ref.shape:
        return {k: "shape mismatch" for k in ("mse", "mse_r", "mse_g", "mse_b", "ssim", "fid_note")}

    # Float64 in [0, 1] for numerical accuracy
    p = pred.astype(np.float64) / 255.0
    r = ref.astype(np.float64) / 255.0

    mse_r = float(np.mean((p[:, :, 0] - r[:, :, 0]) ** 2))
    mse_g = float(np.mean((p[:, :, 1] - r[:, :, 1]) ** 2))
    mse_b = float(np.mean((p[:, :, 2] - r[:, :, 2]) ** 2))
    mse   = (mse_r + mse_g + mse_b) / 3.0

    # SSIM over full image (multichannel)
    ssim_val = ssim_fn(p, r, data_range=1.0, channel_axis=2)

    return {
        "mse":      mse,
        "mse_r":    mse_r,
        "mse_g":    mse_g,
        "mse_b":    mse_b,
        "ssim":     float(ssim_val),
        # FID is a dataset-level metric (requires 50+ image pairs and InceptionV3 features).
        # It cannot be meaningfully computed on a single image pair.
        "fid_note": "N/A — FID requires a batch of images (dataset-level metric)",
    }


# ── Inference ──────────────────────────────────────────────────────────────────

def pad_to_multiple(t: torch.Tensor, multiple: int = 16):
    """
    Pad a 1CHW tensor so H and W are divisible by `multiple`.
    Uses reflect padding (bottom/right only) so border pixels mirror real image
    content. Returns (padded_tensor, (h_orig, w_orig)) for exact crop-back.
    """
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, (h, w)


def enhance_image(input_img: np.ndarray) -> np.ndarray:
    """
    Run U-Net enhancement at full resolution.
    Falls back to gamma brightening if model failed to load.
    """
    if input_img is None:
        return None

    model, model_loaded = get_model()
    if model_loaded and model is not None:
        t = torch.from_numpy(
            input_img.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)          # 1CHW

        # U-Net has 4 MaxPool layers → H and W must be divisible by 16.
        # Reflect-pad, run inference, then crop back to original dimensions.
        t, (h_orig, w_orig) = pad_to_multiple(t, multiple=16)
        with torch.no_grad():
            out = model(t)
        out = out[:, :, :h_orig, :w_orig]

        result = out.squeeze(0).permute(1, 2, 0).numpy()  # HWC
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    else:
        # Fallback: gamma brightening
        arr = input_img.astype(np.float32) / 255.0
        brightened = np.power(np.clip(arr, 0, 1), 0.5)
        return (brightened * 255).astype(np.uint8)


# ── Main handler ───────────────────────────────────────────────────────────────

def run(input_img, ref_img):
    """
    Called by Gradio on submit.
    Returns: enhanced_img, mse_str, mse_r_str, mse_g_str, mse_b_str, ssim_str, fid_str, status_str
    """
    if input_img is None:
        empty = ("—",) * 6
        return None, *empty, "⚠️ Please upload a low-light input image."

    enhanced = enhance_image(input_img)

    _, model_loaded = get_model()
    model_status = (
        "✅ U-Net loaded (epoch 22, val MSE 0.0290) — tyakovenko/night-to-day-enhancement-model"
        if model_loaded
        else "⚠️ Model unavailable — showing gamma-brightened placeholder"
    )

    if ref_img is None:
        return enhanced, "—", "—", "—", "—", "—", "—", model_status + " | Upload reference image to compute metrics."

    metrics = compute_metrics(enhanced, ref_img)

    def fmt(v):
        return f"{v:.6f}" if isinstance(v, float) else str(v)

    return (
        enhanced,
        fmt(metrics["mse"]),
        fmt(metrics["mse_r"]),
        fmt(metrics["mse_g"]),
        fmt(metrics["mse_b"]),
        fmt(metrics["ssim"]),
        metrics["fid_note"],
        model_status,
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

CSS = """
#title { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.metric-box { font-family: monospace; font-size: 1.05em; }
#status-bar { font-size: 0.85em; color: #aaa; margin-top: 8px; }
"""

with gr.Blocks(css=CSS, title="Low-Light Enhancement") as demo:

    gr.Markdown("# 🌙 → ☀️  Low-Light Image Enhancement", elem_id="title")
    gr.Markdown(
        "Upload a low-light or night image to enhance it to daylight appearance. "
        "Optionally upload a reference (ground-truth) image to compute evaluation metrics.",
        elem_id="subtitle",
    )

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────────────────────
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="Input — Low-Light Image",
                type="numpy",
                image_mode="RGB",
            )
            ref_img = gr.Image(
                label="Reference — Day Image (optional, for metrics)",
                type="numpy",
                image_mode="RGB",
            )
            enhance_btn = gr.Button("✨ Enhance", variant="primary")

        # ── Right column: output ──────────────────────────────────────────────
        with gr.Column(scale=1):
            output_img = gr.Image(
                label="Output — Enhanced Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
            )

    # ── Metrics row ───────────────────────────────────────────────────────────
    gr.Markdown("### Evaluation Metrics")
    gr.Markdown(
        "_Requires a reference image. All MSE values are in [0, 1] scale (float64). "
        "SSIM ranges from -1 to 1 (higher = better). "
        "FID is a dataset-level metric and cannot be computed for a single image pair._"
    )

    with gr.Row():
        mse_out    = gr.Textbox(label="MSE (avg)",        elem_classes="metric-box", interactive=False)
        ssim_out   = gr.Textbox(label="SSIM",             elem_classes="metric-box", interactive=False)
        fid_out    = gr.Textbox(label="FID",              elem_classes="metric-box", interactive=False)

    with gr.Row():
        mse_r_out  = gr.Textbox(label="MSE — R channel",  elem_classes="metric-box", interactive=False)
        mse_g_out  = gr.Textbox(label="MSE — G channel",  elem_classes="metric-box", interactive=False)
        mse_b_out  = gr.Textbox(label="MSE — B channel",  elem_classes="metric-box", interactive=False)

    status_bar = gr.Markdown("", elem_id="status-bar")

    # ── Wire up ───────────────────────────────────────────────────────────────
    enhance_btn.click(
        fn=run,
        inputs=[input_img, ref_img],
        outputs=[output_img, mse_out, mse_r_out, mse_g_out, mse_b_out, ssim_out, fid_out, status_bar],
    )

    # ── Examples ─────────────────────────────────────────────────────────────
    gr.Markdown("### Examples")
    gr.Markdown(
        "_Upload your own image above, or use the final evaluation pair "
        "(`night.jpg` / `day.jpg`) from the repo root._"
    )


demo.launch()
