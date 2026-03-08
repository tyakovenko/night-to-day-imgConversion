"""
Gradio UI — Low-Light to Day Image Enhancement
Hosted at: huggingface.co/spaces/tyakovenko/night-to-day-enhancement

UI skeleton — model connection is a placeholder until training completes.
Once checkpoints/best.pt is available (or uploaded to HF Hub model repo),
replace MODEL_REPO_ID / load_model() with the real loader.
"""

import numpy as np
from PIL import Image
import gradio as gr
from skimage.metrics import structural_similarity as ssim_fn

# ── Model placeholder ──────────────────────────────────────────────────────────
# TODO: replace with real checkpoint loader after training completes.
# The model checkpoint will be stored at:
#   HF Hub model repo: tyakovenko/night-to-day-enhancement-model
# Load with:
#   from huggingface_hub import hf_hub_download
#   from model import UNet
#   ckpt_path = hf_hub_download("tyakovenko/night-to-day-enhancement-model", "best.pt")
#   model = UNet(base_filters=16); model.load_state_dict(torch.load(ckpt_path)["model"])

MODEL_LOADED = False  # flip to True once checkpoint is wired in

def load_model():
    """Stub — returns None until checkpoint is available."""
    return None

_model = load_model()


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

def enhance_image(input_img: np.ndarray) -> np.ndarray:
    """
    Run enhancement on a single image.
    Skeleton: returns a brightened version until the model is wired in.
    Replace the body of this function with real U-Net inference post-training.
    """
    if input_img is None:
        return None

    if MODEL_LOADED and _model is not None:
        # Real inference path (activated post-training)
        import torch
        from model import UNet
        _model.eval()
        t = torch.from_numpy(
            input_img.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = _model(t)
        result = out.squeeze(0).permute(1, 2, 0).numpy()
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    else:
        # Placeholder: simple gamma brightening to show the pipeline is wired
        arr = input_img.astype(np.float32) / 255.0
        brightened = np.power(arr, 0.5)          # gamma = 0.5 → brightens
        brightened = np.clip(brightened, 0, 1)
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

    model_status = (
        "✅ Model loaded — U-Net (base_filters=16)"
        if MODEL_LOADED
        else "⚠️ Model not yet connected — showing gamma-brightened placeholder"
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
        "(`night.png` / `day.png`) in the repo root once available._"
    )


if __name__ == "__main__":
    demo.launch()
