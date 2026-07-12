"""
七种文本→动作路径的全景概览图

展示 starVLA 中文本模态影响动作生成的七种技术路径，
从左到右按「VLM 参与深度」排列。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#2980b9"
RED = "#c0392b"
GREEN = "#27ae60"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
TEAL = "#16a085"
PINK = "#e84393"
GRAY = "#7f8c8d"
DARK = "#2c3e50"
WHITE = "#ffffff"

fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(-0.5, 21)
ax.set_ylim(-1, 15)
ax.axis("off")

def draw_box(x, y, w, h, label, color, fontsize=8, textcolor="white", alpha=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08", facecolor=color, edgecolor=DARK,
                          linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4, wrap=True)

def draw_arrow(x1, y1, x2, y2, color=DARK, lw=1.5, style="-|>"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, lw=lw, mutation_scale=12, zorder=2)
    ax.add_patch(arrow)

# ── Title ──
ax.text(10.5, 14.5, "starVLA: Seven Pathways from Text to Action",
        ha="center", va="center", fontsize=16, fontweight="bold", color=DARK)
ax.text(10.5, 14.0, "Sorted by text-action coupling tightness (loose → tight)",
        ha="center", va="center", fontsize=10, color=GRAY, style="italic")

# ── Common top row: shared input ──
draw_box(4.5, 13, 3.5, 0.65, "Task Instruction (lang)", BLUE, 9)
draw_box(10.5, 13, 3.5, 0.65, "Images (multi-view)", TEAL, 9)
draw_box(16, 13, 3.5, 0.65, "Robot State (optional)", ORANGE, 9)

# ── Column definitions ──
cols = [
    {"x": 1.2, "label": "Type 1\nCross-Attn", "sub": "QwenGR00T\nCosmosGR00T\nQwenDual", "color": BLUE},
    {"x": 4.2, "label": "Type 2\nLayerwise\nCross-Attn", "sub": "QwenPI\nQwenPI_v3", "color": PURPLE},
    {"x": 7.2, "label": "Type 3\nConcat+CFG", "sub": "M1\n(InternVLA)", "color": RED},
    {"x": 10.2, "label": "Type 4\nAction Token\nExtraction", "sub": "QwenOFT\nQwenAdapter", "color": GREEN},
    {"x": 13.2, "label": "Type 5\nAutoregressive\nDiscrete", "sub": "QwenFast", "color": ORANGE},
    {"x": 16.2, "label": "Type 6\nDual-Branch\nLLR", "sub": "LangForce", "color": PINK},
    {"x": 19.2, "label": "Type 7\nShared-Layer\nFusion", "sub": "PI0 / PI0.5", "color": TEAL},
]

for col in cols:
    cx = col["x"]

    # Type label
    draw_box(cx, 11.5, 2.3, 1.2, col["label"], col["color"], 7)

    # Framework names
    draw_box(cx, 10, 2.3, 0.7, col["sub"], col["color"], 6, alpha=0.7)

    # Arrow from input to type
    draw_arrow(4.5, 12.65, cx, 12.1, color=BLUE, lw=0.8)

# ── Row 2: VLM processing ──
vlm_labels = [
    "VLM\nhidden[-1]",
    "VLM\nhidden[-N:]",
    "VLM layers\n[s:e]+QFormer",
    "VLM\n+ placeholder\ntokens",
    "VLM\ngenerate()",
    "VLM\ndual-branch\nP(A+L) / P(L+A)",
    "VLM\n+ Action Expert\nshared attn",
]
for i, (col, vlm_lbl) in enumerate(zip(cols, vlm_labels)):
    draw_box(col["x"], 8.5, 2.3, 1.1, vlm_lbl, GRAY, 6, WHITE)
    draw_arrow(col["x"], 9.65, col["x"], 9.1, color=DARK, lw=1)

# ── Row 3: Conditioning mechanism ──
cond_labels = [
    "Cross-Attn\nDiT",
    "Layerwise\nCross-Attn\nDiT",
    "Concat\nSelf-Attn\nDiT + CFG",
    "Gather\nhidden @\nplaceholder",
    "Extract\naction tokens\n→ FAST decode",
    "Extract\naction query\nhidden",
    "Joint Q/K/V\nattention\nfusion",
]
cond_colors = [BLUE, PURPLE, RED, GREEN, ORANGE, PINK, TEAL]
for i, (col, cond_lbl, cc) in enumerate(zip(cols, cond_labels, cond_colors)):
    draw_box(col["x"], 6.8, 2.3, 1.1, cond_lbl, cc, 6, alpha=0.85)
    draw_arrow(col["x"], 7.95, col["x"], 7.4, color=DARK, lw=1)

# ── Row 4: Action head ──
head_labels = [
    "Flow\nMatching",
    "Flow\nMatching",
    "DDPM\nDiffusion",
    "MLP\nL1 Loss",
    "(VLM is\nthe head)",
    "Flow\nMatching",
    "Flow\nMatching",
]
for i, (col, head_lbl) in enumerate(zip(cols, head_labels)):
    draw_box(col["x"], 5.3, 2.1, 0.7, head_lbl, DARK, 7, WHITE)
    draw_arrow(col["x"], 6.25, col["x"], 5.7, color=DARK, lw=1)

# ── Row 5: Output ──
for col in cols:
    draw_box(col["x"], 4.1, 2.1, 0.6, "Continuous\nActions", "#1abc9c", 7)
    draw_arrow(col["x"], 4.95, col["x"], 4.45, color=DARK, lw=1)

# ── Row 6: Loss type ──
loss_labels = [
    "MSE\nvelocity",
    "MSE\nvelocity",
    "MSE\nnoise + CFG\ndropout",
    "L1",
    "CE\n(next-token)",
    "FM + LLR\n(dual loss)",
    "MSE\nvelocity",
]
for i, (col, loss_lbl) in enumerate(zip(cols, loss_labels)):
    draw_box(col["x"], 3.0, 2.1, 0.7, loss_lbl, "#34495e", 6, WHITE, alpha=0.8)
    draw_arrow(col["x"], 3.75, col["x"], 3.4, color=DARK, lw=0.8)

# ── Legend boxes at bottom ──
legend_items = [
    ("Text conditioning tightness", GRAY),
]
ax.annotate("", xy=(19.5, 1.8), xytext=(1.0, 1.8),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2))
ax.text(10.25, 1.4, "Loose coupling ──────────────────────────── Tight coupling",
        ha="center", va="center", fontsize=10, color=DARK, style="italic")
ax.text(1.0, 1.0, "Text only conditions\nhidden representations",
        ha="center", va="center", fontsize=7, color=BLUE)
ax.text(19.5, 1.0, "Text directly participates\nin action token generation",
        ha="center", va="center", fontsize=7, color=ORANGE)

# ── Highlight CFG ──
cfg_box = FancyBboxPatch((6.0, 4.7), 2.4, 8.0,
                          boxstyle="round,pad=0.15", facecolor="none", edgecolor=RED,
                          linewidth=2.5, linestyle="--", zorder=1)
ax.add_patch(cfg_box)
ax.text(7.2, 12.9, "CFG", fontsize=10, fontweight="bold", color=RED,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fadbd8", edgecolor=RED))

plt.tight_layout()
plt.savefig("/home/luogang/SRC/Robot/starVLA/b/d/asset/text_action_pathways.png", dpi=180, bbox_inches="tight")
print("Saved text_action_pathways.png")
