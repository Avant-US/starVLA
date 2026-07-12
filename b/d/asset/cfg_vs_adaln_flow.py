"""
CFG vs AdaNorm 在模型中的数据流对比图

左侧：CFG (Classifier-Free Guidance) 在 M1 中的数据流
  - 训练时 10% dropout，推理时 double batch + guidance 公式
右侧：AdaLN (Adaptive Layer Normalization) 在 NVIDIA DiT 中的数据流
  - 每层 adaLN 注入 timestep，cross-attention 注入 VLM features
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

BLUE = "#2980b9"
RED = "#c0392b"
GREEN = "#27ae60"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
TEAL = "#16a085"
GRAY = "#7f8c8d"
DARK = "#2c3e50"
GOLD = "#f1c40f"


def draw_box(ax, x, y, w, h, label, color, fontsize=9, textcolor="white", alpha=1.0, ls="-"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08", facecolor=color, edgecolor=DARK,
        linewidth=1.5 if ls == "-" else 2.0, alpha=alpha, zorder=3,
        linestyle=ls
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.8, style="-|>"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw, mutation_scale=14, zorder=2
    )
    ax.add_patch(arrow)


# ══════════════════════════════════════════════════════════
# LEFT PANEL: CFG in M1 (Facebook DiT)
# ══════════════════════════════════════════════════════════
ax = ax1
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 17)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 16.5, "CFG (Classifier-Free Guidance)",
        ha="center", fontsize=14, fontweight="bold", color=RED)
ax.text(5, 16.0, "M1 Framework / Facebook DiT",
        ha="center", fontsize=10, color=GRAY, style="italic")

# ── Training Path (left sub) ──
ax.text(2.5, 15.2, "TRAINING", ha="center", fontsize=10,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fadbd8", edgecolor=RED))

draw_box(ax, 2.5, 14.0, 4, 0.7, "Condition z (VLM+DINO)", BLUE, 8)
draw_box(ax, 2.5, 12.8, 4, 0.8, "LabelEmbedder\ntoken_drop(p=0.1)", RED, 7)
ax.text(5.2, 12.8, "10% replaced\nby learned\nuncondition",
        fontsize=6, color=RED, style="italic")
draw_box(ax, 2.5, 11.5, 4, 0.7, "c = t_emb + z_emb", PURPLE, 8)
draw_box(ax, 2.5, 10.3, 4, 0.7, "Concat [c; x_noisy]", GRAY, 8, DARK)
draw_box(ax, 2.5, 9.1, 4, 0.8, "Self-Attention DiT\n(plain LayerNorm)", DARK, 7)
draw_box(ax, 2.5, 7.9, 4, 0.7, "MSE noise loss", GREEN, 8)

draw_arrow(ax, 2.5, 13.65, 2.5, 13.2)
draw_arrow(ax, 2.5, 12.4, 2.5, 11.85)
draw_arrow(ax, 2.5, 11.15, 2.5, 10.65)
draw_arrow(ax, 2.5, 9.95, 2.5, 9.5)
draw_arrow(ax, 2.5, 8.7, 2.5, 8.25)

# ── Inference Path (right sub) ──
ax.text(7.8, 15.2, "INFERENCE", ha="center", fontsize=10,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#d5f5e3", edgecolor=GREEN))

draw_box(ax, 7.8, 14.0, 3.5, 0.7, "Condition z", BLUE, 8)
draw_box(ax, 7.8, 12.8, 3.5, 0.7, "Uncondition (learned)", RED, 7, ls="--")

draw_box(ax, 7.8, 11.3, 3.5, 0.8, "Double batch:\nz_cat = [z; uncond]", ORANGE, 7)
draw_arrow(ax, 7.8, 13.65, 7.8, 13.2, color=BLUE)
draw_arrow(ax, 7.8, 12.45, 7.8, 11.7, color=RED)
draw_arrow(ax, 7.8, 13.65, 7.8, 11.7, color=BLUE)

draw_box(ax, 7.8, 9.9, 3.5, 0.8, "DiT forward\n(both halves)", DARK, 7)
draw_arrow(ax, 7.8, 10.9, 7.8, 10.3)

draw_box(ax, 7.8, 8.5, 3.5, 0.8, "Split output:\ncond / uncond", PURPLE, 7)
draw_arrow(ax, 7.8, 9.5, 7.8, 8.9)

draw_box(ax, 7.8, 7.0, 3.5, 0.9, "CFG formula:\nout = uncond +\n  s*(cond - uncond)", RED, 6)
draw_arrow(ax, 7.8, 8.1, 7.8, 7.45)

draw_box(ax, 7.8, 5.6, 3.5, 0.7, "DDIM sample loop", GREEN, 8)
draw_arrow(ax, 7.8, 6.55, 7.8, 5.95)

# ── Key characteristics ──
chars_y = 4.2
ax.text(5, chars_y + 0.6, "Key Characteristics", ha="center", fontsize=10,
        fontweight="bold", color=RED)
char_items = [
    "No adaLN - uses plain LayerNorm",
    "Condition injected via token concatenation",
    "Inference cost: 2x forward pass",
    "Guidance strength s tunable at inference",
    "Only in M1 (1 framework)",
]
for i, item in enumerate(char_items):
    ax.text(1.0, chars_y - i * 0.45, f"  {item}",
            fontsize=7, color=DARK, va="center")
    ax.plot(0.6, chars_y - i * 0.45, "s", color=RED, markersize=5)

# ══════════════════════════════════════════════════════════
# RIGHT PANEL: AdaLN in NVIDIA DiT
# ══════════════════════════════════════════════════════════
ax = ax2
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 17)
ax.set_aspect("equal")
ax.axis("off")

ax.text(5, 16.5, "AdaLN (Adaptive Layer Normalization)",
        ha="center", fontsize=14, fontweight="bold", color=BLUE)
ax.text(5, 16.0, "NVIDIA DiT / 14+ Frameworks",
        ha="center", fontsize=10, color=GRAY, style="italic")

# ── Both Training & Inference ──
ax.text(5, 15.2, "TRAINING & INFERENCE (identical path)",
        ha="center", fontsize=10, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#d6eaf8", edgecolor=BLUE))

# Input row
draw_box(ax, 2.5, 14.0, 3.5, 0.7, "VLM Hidden States", BLUE, 8)
draw_box(ax, 7.5, 14.0, 3.5, 0.7, "Timestep t", ORANGE, 8)

# Timestep encoder
draw_box(ax, 7.5, 12.8, 3.5, 0.7, "TimestepEncoder", ORANGE, 8)
draw_arrow(ax, 7.5, 13.65, 7.5, 13.15)

# DiT Block (big box)
block_y = 10.2
block_h = 4.2
block = FancyBboxPatch(
    (0.5, block_y - block_h / 2), 9.0, block_h,
    boxstyle="round,pad=0.15", facecolor="#eaf2f8", edgecolor=BLUE,
    linewidth=2.0, zorder=1
)
ax.add_patch(block)
ax.text(5, block_y + block_h / 2 - 0.2, "BasicTransformerBlock (x N layers)",
        ha="center", fontsize=9, fontweight="bold", color=BLUE)

# adaLN inside block
draw_box(ax, 7, 11.5, 2.8, 0.7, "adaLN\nscale, shift", ORANGE, 7)
draw_arrow(ax, 7.5, 12.45, 7, 11.85, color=ORANGE, lw=2.5)
ax.text(9.5, 11.5, r"$\mathrm{LN}(x) \cdot$" + "\n" + r"$(1+s) + d$",
        fontsize=7, color=ORANGE, ha="center", style="italic")

# Self-attention
draw_box(ax, 3.5, 11.0, 2.5, 0.6, "Self-Attention", TEAL, 7)
draw_arrow(ax, 5.6, 11.5, 4.75, 11.15, color=ORANGE, lw=1.5)

# Cross-attention
draw_box(ax, 3.5, 10.0, 2.5, 0.6, "Cross-Attention", PURPLE, 7)
draw_arrow(ax, 2.5, 13.65, 2.5, 10.3, color=BLUE, lw=2.0)
ax.text(1.0, 11.5, "K, V from\nVLM", fontsize=7, color=BLUE, ha="center", style="italic")

# FFN
draw_box(ax, 3.5, 9.0, 2.5, 0.6, "Feed-Forward", GREEN, 7)
draw_arrow(ax, 3.5, 10.65, 3.5, 10.3, color=DARK, lw=1)
draw_arrow(ax, 3.5, 9.7, 3.5, 9.3, color=DARK, lw=1)

# Output adaLN
draw_box(ax, 5, 7.2, 4.0, 0.8, "Output adaLN\nnorm_out * (1+s) + d", ORANGE, 7)
draw_arrow(ax, 3.5, 8.7, 5, 7.6, color=DARK, lw=1.5)
draw_arrow(ax, 7.5, 12.45, 7.5, 7.6, color=ORANGE, lw=1.5)

# Projection
draw_box(ax, 5, 6.0, 4.0, 0.6, "proj_out_2 (Linear)", DARK, 8)
draw_arrow(ax, 5, 6.8, 5, 6.3)

# Action output
draw_box(ax, 5, 5.0, 3.0, 0.6, "Velocity prediction", GREEN, 8)
draw_arrow(ax, 5, 5.7, 5, 5.3)

# Flow matching sampling
draw_box(ax, 5, 4.0, 3.5, 0.7, "Euler ODE sampling", TEAL, 8)
draw_arrow(ax, 5, 4.7, 5, 4.35)

# ── Key characteristics ──
chars_y = 2.6
ax.text(5, chars_y + 0.6, "Key Characteristics", ha="center", fontsize=10,
        fontweight="bold", color=BLUE)
char_items = [
    "adaLN injects timestep at every layer",
    "VLM features via cross-attention (separate)",
    "Inference cost: 1x forward per step",
    "No tunable guidance strength",
    "Used by 14+ frameworks (dominant)",
]
for i, item in enumerate(char_items):
    ax.text(1.0, chars_y - i * 0.45, f"  {item}",
            fontsize=7, color=DARK, va="center")
    ax.plot(0.6, chars_y - i * 0.45, "o", color=BLUE, markersize=5)

fig.tight_layout(pad=2.0)

# ── Bottom comparison formula bar ──
fig.text(0.25, 0.01,
         r"CFG: $\hat{\epsilon} = \epsilon_\theta(x_t, \varnothing) + s \cdot [\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)]$",
         ha="center", fontsize=11, color=RED, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor=RED))
fig.text(0.75, 0.01,
         r"adaLN: $x' = \mathrm{LN}(x) \cdot (1 + \gamma(t)) + \beta(t)$",
         ha="center", fontsize=11, color=BLUE, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#d6eaf8", edgecolor=BLUE))

plt.savefig("/home/luogang/SRC/Robot/starVLA/b/d/asset/cfg_vs_adaln_flow.png",
            dpi=180, bbox_inches="tight")
print("Saved cfg_vs_adaln_flow.png")
