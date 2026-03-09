"""
Gradio UI — Low-Light to Day Image Enhancement
Hosted at: huggingface.co/spaces/tyakovenko/night-to-day-enhancement
"""

import numpy as np
from PIL import Image
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from skimage.metrics import structural_similarity as ssim_fn
from model import UNet
from dataset import _compute_global_stats

# ── Model registry ─────────────────────────────────────────────────────────────
# Each entry: (repo_id, filename, residual, global_context)
# residual=True       → UNet predicts a Tanh delta added to input
# global_context=True → UNet uses GlobalContextEncoder; inference computes
#                       per-channel (mean, std, p10) from the full input image

MODEL_OPTIONS = {
    "v1 — best.pt  (Transient Attributes, MSE)":                    ("tyakovenko/night-to-day-enhancement-model",    "best.pt",          False, False),
    "v1-extended — best_extended.pt  (TA + LOL, MSE)":              ("tyakovenko/night-to-day-enhancement-model",    "best_extended.pt", False, False),
    "v2 — best_v2.pt  (TA + LOL, L1 + MS-SSIM)":                   ("tyakovenko/night-to-day-enhancement-model-v2", "best_v2.pt",       False, False),
    "v3 — best_v3.pt  (TA + LOL, Residual + ColorLoss)":            ("tyakovenko/night-to-day-enhancement-model-v3", "best_v3.pt",       True,  False),
    "v4 — best_v4.pt  (TA + LOL, WeightedL1 + LogL1 + GlobalCtx)": ("tyakovenko/night-to-day-enhancement-model-v4", "best_v4.pt",       True,  True),
}
DEFAULT_MODEL = "v4 — best_v4.pt  (TA + LOL, WeightedL1 + LogL1 + GlobalCtx)"

# Cache loaded models by display name so switching is instant after first load
_model_cache: dict = {}  # name → model | None


def get_model(model_name: str):
    """Load model on first call for a given name; return cached instance after."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    repo_id, filename, residual, global_context = MODEL_OPTIONS[model_name]
    try:
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        base_filters = ckpt.get("args", {}).get("base_filters", 16)
        model = UNet(base_filters=base_filters, residual=residual,
                     use_global_context=global_context)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded {filename} from {repo_id} "
              f"(epoch {ckpt.get('epoch','?')}, val MSE {ckpt.get('val_mse', 0):.4f}, "
              f"residual={residual}, global_context={global_context})")
        _model_cache[model_name] = model
    except Exception as e:
        print(f"Model load failed [{model_name}]: {e}")
        _model_cache[model_name] = None

    return _model_cache[model_name]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """
    Channel-wise MSE and SSIM between enhanced image and reference.

    Args:
        pred: HWC uint8 enhanced image
        ref:  HWC uint8 reference (day) image — must match pred shape
    """
    if pred.shape != ref.shape:
        return {k: "shape mismatch" for k in ("mse", "mse_r", "mse_g", "mse_b", "ssim")}

    # Float64 in [0, 1] for numerical accuracy
    p = pred.astype(np.float64) / 255.0
    r = ref.astype(np.float64) / 255.0

    mse_r = float(np.mean((p[:, :, 0] - r[:, :, 0]) ** 2))
    mse_g = float(np.mean((p[:, :, 1] - r[:, :, 1]) ** 2))
    mse_b = float(np.mean((p[:, :, 2] - r[:, :, 2]) ** 2))
    mse   = (mse_r + mse_g + mse_b) / 3.0

    ssim_val = ssim_fn(p, r, data_range=1.0, channel_axis=2)

    return {
        "mse":   mse,
        "mse_r": mse_r,
        "mse_g": mse_g,
        "mse_b": mse_b,
        "ssim":  float(ssim_val),
    }


# ── Inference ──────────────────────────────────────────────────────────────────

def pad_to_multiple(t: torch.Tensor, multiple: int = 16):
    """
    Pad a 1CHW tensor so H and W are divisible by `multiple`.
    Reflect-pads bottom/right only; returns (padded_tensor, (h_orig, w_orig)).
    """
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, (h, w)


def enhance_image(input_img: np.ndarray, model_name: str) -> tuple:
    """
    Run U-Net enhancement at full resolution.
    Falls back to gamma brightening if model failed to load.
    Returns (enhanced_np, status_str).
    """
    if input_img is None:
        return None, "No input image."

    model = get_model(model_name)

    if model is not None:
        img_f32 = input_img.astype(np.float32) / 255.0
        t = torch.from_numpy(img_f32).permute(2, 0, 1).unsqueeze(0)  # 1CHW

        # Compute global stats if this model uses GlobalContextEncoder
        _, _, _, global_context = MODEL_OPTIONS[model_name]
        global_stats_t = None
        if global_context:
            stats = _compute_global_stats(img_f32)
            global_stats_t = torch.from_numpy(stats).unsqueeze(0)  # [1, 9]

        t, (h_orig, w_orig) = pad_to_multiple(t, multiple=16)
        with torch.no_grad():
            out = model(t, global_stats=global_stats_t)
        out = out[:, :, :h_orig, :w_orig]

        result = out.squeeze(0).permute(1, 2, 0).numpy()
        enhanced = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        status = f"✅ {model_name}"
    else:
        arr = input_img.astype(np.float32) / 255.0
        enhanced = (np.power(np.clip(arr, 0, 1), 0.5) * 255).astype(np.uint8)
        status = f"⚠️ Model unavailable ({model_name}) — showing gamma-brightened placeholder"

    return enhanced, status


# ── Main handler ───────────────────────────────────────────────────────────────

def run(input_img, ref_img, model_name):
    """Called by Gradio on submit."""
    if input_img is None:
        return None, "—", "—", "—", "—", "—", "⚠️ Please upload a low-light input image."

    enhanced, status = enhance_image(input_img, model_name)

    if ref_img is None:
        return enhanced, "—", "—", "—", "—", "—", status + " | Upload a reference image to compute metrics."

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
        status,
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

CSS = """
#title    { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; }
.metric-box { font-family: monospace; font-size: 1.05em; }
#status-bar { font-size: 0.85em; color: #aaa; margin-top: 8px; }
#ref-col { display: flex; align-items: flex-end; }
"""

with gr.Blocks(title="Low-Light Enhancement") as demo:

    gr.Markdown("# 🌙 → ☀️  Low-Light Image Enhancement", elem_id="title")
    gr.Markdown(
        "Upload a low-light or night image to enhance it to daylight appearance. "
        "Optionally upload a reference (ground-truth) image to compute evaluation metrics.",
        elem_id="subtitle",
    )

    # ── Controls row: dropdown + button ───────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=DEFAULT_MODEL,
                label="Model",
            )
        with gr.Column(scale=1, min_width=120):
            enhance_btn = gr.Button("✨ Enhance", variant="primary")

    status_bar = gr.Markdown("", elem_id="status-bar")

    # ── Main images row: input and output at the same horizontal level ─────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_img = gr.Image(
                label="Input — Low-Light Image",
                type="numpy",
                image_mode="RGB",
            )
        with gr.Column(scale=1):
            output_img = gr.Image(
                label="Output — Enhanced Image",
                type="numpy",
                image_mode="RGB",
                interactive=False,
            )

    # ── Reference image (optional, below input) ───────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            ref_img = gr.Image(
                label="Reference — Day Image (optional, for metrics)",
                type="numpy",
                image_mode="RGB",
            )
        with gr.Column(scale=1):
            pass  # keeps reference left-aligned, mirroring the input column

    # ── Metrics ───────────────────────────────────────────────────────────────
    gr.Markdown("### Evaluation Metrics")
    gr.Markdown(
        "_Requires a reference image. MSE values are in [0, 1] scale (float64). "
        "SSIM ranges from -1 to 1 (higher is better)._"
    )

    with gr.Row():
        mse_out   = gr.Textbox(label="MSE (avg)",       elem_classes="metric-box", interactive=False)
        ssim_out  = gr.Textbox(label="SSIM",            elem_classes="metric-box", interactive=False)

    with gr.Row():
        mse_r_out = gr.Textbox(label="MSE — R channel", elem_classes="metric-box", interactive=False)
        mse_g_out = gr.Textbox(label="MSE — G channel", elem_classes="metric-box", interactive=False)
        mse_b_out = gr.Textbox(label="MSE — B channel", elem_classes="metric-box", interactive=False)

    # ── Wire up ───────────────────────────────────────────────────────────────
    enhance_btn.click(
        fn=run,
        inputs=[input_img, ref_img, model_dropdown],
        outputs=[output_img, mse_out, mse_r_out, mse_g_out, mse_b_out, ssim_out, status_bar],
    )


demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
