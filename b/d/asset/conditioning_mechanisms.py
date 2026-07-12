"""
三种条件注入机制的结构对比图

展示 Cross-Attention、Token Concatenation + Self-Attention、adaLN
三种将 VLM 特征注入 DiT 的方式。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

BLUE = "#3498db"
RED = "#e74c3c"
GREEN = "#2ecc71"
ORANGE = "#f39c12"
PURPLE = "#9b59b6"
GRAY = "#95a5a6"
DARK = "#2c3e50"
LIGHT_BLUE = "#d6eaf8"
LIGHT_RED = "#fadbd8"
LIGHT_GREEN = "#d5f5e3"
LIGHT_ORANGE = "#fdebd0"
LIGHT_PURPLE = "#e8daef"

def draw_box(ax, x, y, w, h, label, color, fontsize=9, textcolor="white"):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.05", facecolor=color, edgecolor=DARK, linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=DARK, style="-|>", lw=1.5):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, lw=lw, mutation_scale=15, zorder=2)
    ax.add_patch(arrow)

# ── Panel A: Cross-Attention (NVIDIA DiT) ──
ax = axes[0]
ax.set_xlim(-1, 7)
ax.set_ylim(-0.5, 9)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) Cross-Attention\n(NVIDIA DiT / GR00T)", fontsize=12, fontweight="bold", pad=10)

draw_box(ax, 3, 8.2, 4.5, 0.7, "VLM Hidden States", BLUE, 9)
draw_box(ax, 1.2, 6.5, 1.8, 0.7, "K, V", BLUE, 9)
draw_box(ax, 4.8, 6.5, 1.8, 0.7, "Q", RED, 9)
draw_box(ax, 3, 5, 4.5, 0.7, "Cross-Attention", PURPLE, 9)
draw_box(ax, 3, 3.5, 4.5, 0.7, "Feed-Forward", GREEN, 9)
draw_box(ax, 3, 2, 4.5, 0.7, "Action Tokens (noisy)", RED, 9)

draw_box(ax, 3, 0.5, 2.5, 0.6, "Timestep t", ORANGE, 8)
draw_box(ax, 5.5, 0.5, 1.5, 0.6, "adaLN", ORANGE, 8)

draw_arrow(ax, 3, 7.85, 1.2, 6.85)
draw_arrow(ax, 3, 7.85, 4.8, 6.85)
draw_arrow(ax, 1.2, 6.15, 3, 5.35)
draw_arrow(ax, 4.8, 6.15, 3, 5.35)
draw_arrow(ax, 3, 4.65, 3, 3.85)
draw_arrow(ax, 3, 2.35, 3, 4.65, style="-|>", color=RED)
draw_arrow(ax, 4.25, 0.5, 5.0, 0.5)
draw_arrow(ax, 5.5, 0.8, 5.5, 4.65, color=ORANGE)

ax.text(0.2, 5, "VLM\nfeatures", fontsize=7, color=BLUE, ha="center", style="italic")
ax.text(5.8, 5, "Action\ntokens", fontsize=7, color=RED, ha="center", style="italic")

# ── Panel B: Token Concatenation + Self-Attention (Facebook DiT / M1) ──
ax = axes[1]
ax.set_xlim(-1, 7)
ax.set_ylim(-0.5, 9)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(b) Token Concatenation\n(Facebook DiT / M1 + CFG)", fontsize=12, fontweight="bold", pad=10)

draw_box(ax, 1.5, 8.2, 2.5, 0.7, "Condition z", BLUE, 9)
draw_box(ax, 4.8, 8.2, 2, 0.7, "Timestep t", ORANGE, 9)

draw_box(ax, 3, 6.8, 4.5, 0.7, "c = t + z  (elementwise)", PURPLE, 8)

draw_box(ax, 1.5, 5.3, 2, 0.7, "c tokens", BLUE, 9)
draw_box(ax, 4.5, 5.3, 2, 0.7, "x tokens", RED, 9)
draw_box(ax, 3, 4, 4.5, 0.7, "Concat [c; x]", GRAY, 9, DARK)

draw_box(ax, 3, 2.7, 4.5, 0.7, "Self-Attention", PURPLE, 9)
draw_box(ax, 3, 1.3, 4.5, 0.7, "Slice [:, N_c :]", GREEN, 9)
draw_box(ax, 3, 0, 3, 0.6, "Action output", RED, 9)

draw_arrow(ax, 1.5, 7.85, 3, 7.15)
draw_arrow(ax, 4.8, 7.85, 3, 7.15)
draw_arrow(ax, 3, 6.45, 1.5, 5.65)
draw_arrow(ax, 3, 6.45, 4.5, 5.65)
draw_arrow(ax, 1.5, 4.95, 3, 4.35)
draw_arrow(ax, 4.5, 4.95, 3, 4.35)
draw_arrow(ax, 3, 3.65, 3, 3.05)
draw_arrow(ax, 3, 2.35, 3, 1.65)
draw_arrow(ax, 3, 0.95, 3, 0.35)

ax.annotate("CFG: z can be\nreplaced by\nlearned ∅",
            xy=(1.5, 8.2), xytext=(-0.5, 7),
            fontsize=7, color=RED, style="italic",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1))

# ── Panel C: adaRMS (PI0.5) ──
ax = axes[2]
ax.set_xlim(-1, 7)
ax.set_ylim(-0.5, 9)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(c) adaRMS Modulation\n(PI0.5 Action Expert)", fontsize=12, fontweight="bold", pad=10)

draw_box(ax, 3, 8.2, 3.5, 0.7, "Timestep t", ORANGE, 9)
draw_box(ax, 3, 7, 3.5, 0.7, "MLP → adaRMS cond", ORANGE, 8)

for i, (layer_y, label) in enumerate([(5.5, "Layer 1"), (3.5, "Layer 2"), (1.5, "Layer N")]):
    draw_box(ax, 3, layer_y, 4, 1.0, "", LIGHT_PURPLE, 8, DARK)
    ax.text(3, layer_y + 0.25, label, ha="center", va="center", fontsize=8, fontweight="bold", color=DARK)
    ax.text(3, layer_y - 0.2, "RMSNorm · (1+scale) + shift", ha="center", va="center", fontsize=7, color=PURPLE)

    draw_arrow(ax, 5.2, 7, 5.2, layer_y + 0.5, color=ORANGE, lw=1.2)
    ax.plot(5.2, layer_y + 0.15, ">", color=ORANGE, markersize=5, transform=ax.transData)

draw_arrow(ax, 3, 7.85, 3, 7.35)
draw_arrow(ax, 3, 6.65, 3, 6.0)
draw_arrow(ax, 3, 5.0, 3, 4.0)

ax.text(3, 4.5, "...", ha="center", va="center", fontsize=14, color=DARK)
ax.text(3, 2.5, "...", ha="center", va="center", fontsize=14, color=DARK)

draw_arrow(ax, 3, 3.0, 3, 2.0)
draw_arrow(ax, 3, 1.0, 3, 0.3)
draw_box(ax, 3, 0, 3, 0.6, "Action output", RED, 9)

ax.text(5.8, 4.5, "adaRMS\ncond →\neach layer", fontsize=7, color=ORANGE, ha="center", style="italic")

plt.tight_layout(pad=2.0)
plt.savefig("/home/luogang/SRC/Robot/starVLA/b/d/asset/conditioning_mechanisms.png", dpi=180, bbox_inches="tight")
print("Saved conditioning_mechanisms.png")
