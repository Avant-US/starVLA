"""
starVLA Input Modality Taxonomy Diagram

4-row modality processing pipeline:
  Row 1 - Visual Modality (视觉模态)
  Row 2 - Language Modality (语言模态)
  Row 3 - Proprioceptive State (本体感知状态)
  Row 4 - Action Modality (动作模态)

Each row: Raw Input -> Processing / Encoding -> Intermediate Repr -> Consumers
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ── Color Palette ──
BLUE = "#2980b9"
DARK_BLUE = "#1a5276"
GREEN = "#27ae60"
DARK_GREEN = "#1e8449"
ORANGE = "#e67e22"
DARK_ORANGE = "#d35400"
PURPLE = "#8e44ad"
DARK_PURPLE = "#6c3483"
TEAL = "#16a085"
RED = "#c0392b"
GRAY = "#7f8c8d"
DARK = "#2c3e50"
LIGHT_BLUE = "#d6eaf8"
LIGHT_GREEN = "#d5f5e3"
LIGHT_ORANGE = "#fdebd0"
LIGHT_PURPLE = "#e8daef"

# ── Figure Setup ──
fig, ax = plt.subplots(figsize=(28, 20))
ax.set_xlim(-0.5, 28.5)
ax.set_ylim(-0.5, 20.5)
ax.axis("off")

# ── Layout Constants ──
X_RAW = 3.5
W_RAW = 3.8
X_PROC = 11.0
W_PROC = 6.5
X_INTER = 18.5
W_INTER = 4.8
X_CONS = 25.5
W_CONS = 4.2

PROC_H = 0.65
PROC_GAP = 0.10

ROW_CENTERS = [16.5, 11.85, 7.95, 4.15]
ROW_BG_BOUNDS = [(14.2, 18.8), (9.8, 13.9), (6.4, 9.5), (2.2, 6.1)]
ROW_THEMES = [
    (BLUE, DARK_BLUE, LIGHT_BLUE),
    (GREEN, DARK_GREEN, LIGHT_GREEN),
    (ORANGE, DARK_ORANGE, LIGHT_ORANGE),
    (PURPLE, DARK_PURPLE, LIGHT_PURPLE),
]


# ── Helper Functions ──
def draw_box(x, y, w, h, label, facecolor, fontsize=8, textcolor="white", alpha=1.0, edgecolor=None, lw=1.5, family=None):
    ec = edgecolor or DARK
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=facecolor,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    kw = dict(ha="center", va="center", fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
    if family:
        kw["family"] = family
    ax.text(x, y, label, **kw)


def draw_arrow(x1, y1, x2, y2, color=DARK, lw=2.5, style="-|>"):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        color=color,
        lw=lw,
        mutation_scale=15,
        zorder=2,
    )
    ax.add_patch(arrow)


def draw_row_bg(y_bot, y_top, color, alpha=0.07):
    box = FancyBboxPatch(
        (0.2, y_bot),
        28.0,
        y_top - y_bot,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=0,
    )
    ax.add_patch(box)


def stack_boxes(x, yc, labels, w, h, theme, theme_alt, fontsize=7, gap=0.10):
    """Draw vertically stacked sub-boxes centered at yc. Return y-centers."""
    n = len(labels)
    total = n * h + (n - 1) * gap
    top_y = yc + total / 2 - h / 2
    positions = [top_y - i * (h + gap) for i in range(n)]
    for i, (lbl, yp) in enumerate(zip(labels, positions)):
        col = theme if i % 2 == 0 else theme_alt
        draw_box(x, yp, w, h, lbl, col, fontsize=fontsize)
    return positions


def draw_flow_arrows(yc, theme, lw=2.5):
    """Draw three horizontal flow arrows between the four columns."""
    off = 0.15
    draw_arrow(X_RAW + W_RAW / 2 + off, yc, X_PROC - W_PROC / 2 - off, yc, color=theme, lw=lw)
    draw_arrow(X_PROC + W_PROC / 2 + off, yc, X_INTER - W_INTER / 2 - off, yc, color=theme, lw=lw)
    draw_arrow(X_INTER + W_INTER / 2 + off, yc, X_CONS - W_CONS / 2 - off, yc, color=theme, lw=lw)


# ═══════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════
ax.text(
    14,
    20.0,
    "starVLA Input Modality Taxonomy",
    ha="center",
    va="center",
    fontsize=22,
    fontweight="bold",
    color=DARK,
)
ax.text(
    14,
    19.5,
    "Raw Input  →  Processing / Encoding  →  Intermediate Representation  →  Downstream Consumers",
    ha="center",
    va="center",
    fontsize=11,
    color=GRAY,
    fontstyle="italic",
)

# ═══════════════════════════════════════════════════════
# Column Headers (text labels above Row 1)
# ═══════════════════════════════════════════════════════
for cx, label in [
    (X_RAW, "Raw Input"),
    (X_PROC, "Processing / Encoding"),
    (X_INTER, "Intermediate Repr"),
    (X_CONS, "Downstream Consumers"),
]:
    ax.text(cx, 19.05, label, ha="center", va="center", fontsize=9, fontweight="bold", color=DARK)


# ═══════════════════════════════════════════════════════
# Row Data
# ═══════════════════════════════════════════════════════
ROWS = [
    # ── Row 1: Visual Modality ──
    dict(
        name="Visual Modality (视觉模态)",
        raw="Multi-view RGB\nImages\n\n1–8 views\nexterior_image_1/2\nwrist_image",
        raw_h=3.0,
        proc=[
            "Qwen-VL ViT\npatch 14, mrope, hidden=1536/3584",
            "PaliGemma SigLIP\npatch 14, hidden=1152",
            "DINOv2\nvits/vitb/vitl/vitg, 384–1408d",
            "CosmoPredict2 VAE\n48-ch latent, T5 encoder",
            "Wan2.2 VAE\n48-ch latent, UMT5 encoder",
        ],
        inter=["Visual Token Sequence", "Latent Feature Maps"],
        cons=["VLM Layers", "World Model DiT", "Cross-Attention DiT"],
        src="qwen2_5_vl/ | siglip/ | dinov2/ | cosmos_predict2/ | wan2_2/",
    ),
    # ── Row 2: Language Modality ──
    dict(
        name="Language Modality (语言模态)",
        raw="Instruction Text\n\nNatural language\ntask descriptions",
        raw_h=2.5,
        proc=[
            "Qwen AutoProcessor\nchat template, <image> tokens",
            "SentencePiece\nGemma tokenizer",
            "T5 / UMT5 Text Encoder\nhidden states for cross-attn",
            "CoT_prompt Template\nchain-of-thought wrapping",
        ],
        inter=["Text Token Embeddings", "Text Hidden States"],
        cons=["VLM Self-Attention", "WM Cross-Attention"],
        src="qwen_processor/ | gemma_tokenizer/ | t5_encoder/",
    ),
    # ── Row 3: Proprioceptive State ──
    dict(
        name="Proprioceptive State (本体感知状态)",
        raw="Joint Angles\nEEF Pos/Rot\nGripper\n\n7–42 dim vector",
        raw_h=2.2,
        proc=[
            "MLP State Encoder\nLinear→ReLU→Linear (GR00T)",
            "256-bin Discretization\nquantize to tokens (π₀.5)",
            "ProprioProjector\nLinear→GELU→Linear (Adapter)",
        ],
        inter=["State Embedding", "Discrete Token IDs"],
        cons=["DiT Action Head (concat)", "VLM Input Seq (text inject)"],
        src="groot_action/ | pi_action/ | proprio_projector.py",
    ),
    # ── Row 4: Action Modality ──
    dict(
        name="Action Modality (动作模态)",
        raw="Target Actions\n\n7–32 dim\nchunk=1–100 steps",
        raw_h=2.5,
        proc=[
            "Flow-matching ActionEncoder\nLinear+SiLU+Linear, timestep emb",
            "FAST BPE Tokenizer\ncontinuous→discrete 2048 tokens",
            "MLP L1 Regression\ntarget",
            "MaskGIT Binning\nActionBinning 256 bins",
        ],
        inter=["Noisy Action Trajectory", "Action Token IDs", "Binned Categories"],
        cons=["DiT Velocity Prediction", "VLM Next-token Pred", "MLP Head"],
        src="action_encoder.py | fast_tokenizer/ | action_binning.py",
    ),
]


# ═══════════════════════════════════════════════════════
# Draw All Rows
# ═══════════════════════════════════════════════════════
for r_idx, row in enumerate(ROWS):
    yc = ROW_CENTERS[r_idx]
    bg_bot, bg_top = ROW_BG_BOUNDS[r_idx]
    theme, theme_alt, theme_light = ROW_THEMES[r_idx]

    # Background strip
    draw_row_bg(bg_bot, bg_top, theme)

    # Row label (top-left of background strip)
    ax.text(0.5, bg_top - 0.3, row["name"], ha="left", va="center", fontsize=9, fontweight="bold", color=theme)

    # Raw Input box
    draw_box(X_RAW, yc, W_RAW, row["raw_h"], row["raw"], theme, fontsize=7.5)

    # Processing sub-boxes
    proc_yps = stack_boxes(X_PROC, yc, row["proc"], W_PROC, PROC_H, theme, theme_alt, fontsize=6.5)

    # Source file label (below processing column)
    ax.text(
        X_PROC,
        proc_yps[-1] - 0.5,
        row["src"],
        ha="center",
        va="center",
        fontsize=5,
        color=theme,
        family="monospace",
        fontstyle="italic",
    )

    # Intermediate representation sub-boxes
    stack_boxes(X_INTER, yc, row["inter"], W_INTER, 0.60, theme, theme_alt, fontsize=7)

    # Consumer sub-boxes
    stack_boxes(X_CONS, yc, row["cons"], W_CONS, 0.60, theme, theme_alt, fontsize=7)

    # Flow arrows connecting the four columns
    draw_flow_arrows(yc, theme)


# ═══════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════
plt.tight_layout(pad=1.0)
plt.savefig(
    "/home/luogang/SRC/Robot/starVLA/b/d/asset/modality_input_taxonomy.png",
    dpi=180,
    bbox_inches="tight",
)
print("Saved modality_input_taxonomy.png")
