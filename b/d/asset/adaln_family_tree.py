"""
自适应归一化（Adaptive Normalization）族谱图

展示 adaLN、adaRMS、FiLM 三种变体的统一公式和差异。
所有变体都属于 y = f(x) * (1 + γ(c)) + β(c) 的特化形式。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 15)
ax.axis("off")

BLUE = "#2980b9"
RED = "#c0392b"
GREEN = "#27ae60"
ORANGE = "#e67e22"
PURPLE = "#8e44ad"
TEAL = "#16a085"
PINK = "#e84393"
GRAY = "#7f8c8d"
DARK = "#2c3e50"
LIGHT_BLUE = "#d6eaf8"
LIGHT_GREEN = "#d5f5e3"
LIGHT_ORANGE = "#fdebd0"
LIGHT_PURPLE = "#e8daef"


def draw_box(x, y, w, h, label, color, fontsize=9, textcolor="white", alpha=1.0):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor=DARK,
        linewidth=1.5, alpha=alpha, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)


def draw_arrow(x1, y1, x2, y2, color=DARK, lw=2.0, style="-|>"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, lw=lw, mutation_scale=15, zorder=2
    )
    ax.add_patch(arrow)


# ── Title ──
ax.text(10, 14.3, "Adaptive Normalization Family in starVLA",
        ha="center", va="center", fontsize=18, fontweight="bold", color=DARK)
ax.text(10, 13.7, r"Unified formula: $y = f(x) \cdot (1 + \gamma(c)) + \beta(c)$",
        ha="center", va="center", fontsize=13, color=GRAY, style="italic")

# ── Root: Unified FiLM Formula ──
draw_box(10, 12.5, 8, 0.9, r"FiLM family:  $y = f(x) \cdot (1 + \gamma) + \beta$",
         DARK, 11)

# ── Level 1: Three branches ──
branches = [
    {
        "x": 3.5, "label": "AdaLayerNorm\n(NVIDIA DiT)",
        "color": BLUE, "y": 10.5,
        "norm": "f = LayerNorm",
        "outputs": "2 outputs:\nscale, shift",
        "cond": "timestep t",
        "init": "default init",
        "gate": "no gate",
        "file": "cross_attention_dit.py\nL45-68",
        "users": "14 frameworks\n(QwenGR00T, PI, etc.)",
        "status": "ACTIVE (main)",
    },
    {
        "x": 10, "label": "adaRMS\n(PI0.5 Gemma)",
        "color": PURPLE, "y": 10.5,
        "norm": "f = RMSNorm",
        "outputs": "3 outputs:\nscale, shift, gate",
        "cond": "timestep t",
        "init": "zero init",
        "gate": "gated residual",
        "file": "modeling_gemma.py\nL46-75",
        "users": "PI0/PI0.5\n(3 files)",
        "status": "ACTIVE",
    },
    {
        "x": 16.5, "label": "FiLM\n(State Modulator)",
        "color": ORANGE, "y": 10.5,
        "norm": "f = Identity\n(no norm)",
        "outputs": "2 outputs:\ngamma, beta",
        "cond": "robot state",
        "init": "default init",
        "gate": "no gate\n(simple variant)",
        "file": "spike_action_model\n_multitimestep.py",
        "users": "not imported\nby any framework",
        "status": "VESTIGIAL",
    },
]

for br in branches:
    draw_box(br["x"], br["y"], 4.2, 1.0, br["label"], br["color"], 10)
    draw_arrow(10, 12.05, br["x"], 11.0, color=br["color"], lw=2.5)

# ── Level 2: Detail cards ──
card_y_start = 8.8
card_h = 0.55
card_gap = 0.7

detail_labels = ["norm", "outputs", "cond", "init", "gate"]
detail_titles = ["Norm fn", "Outputs", "Condition", "Init", "Gate"]
detail_colors = [LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE, LIGHT_PURPLE, "#fce4ec"]

for i, (key, title, bg) in enumerate(zip(detail_labels, detail_titles, detail_colors)):
    y = card_y_start - i * card_gap

    # Row label
    ax.text(-0.3, y, title, ha="right", va="center", fontsize=8,
            fontweight="bold", color=DARK)

    for br in branches:
        box = FancyBboxPatch(
            (br["x"] - 2.0, y - card_h / 2), 4.0, card_h,
            boxstyle="round,pad=0.05", facecolor=bg, edgecolor=GRAY,
            linewidth=0.8, alpha=0.8, zorder=3
        )
        ax.add_patch(box)
        ax.text(br["x"], y, br[key], ha="center", va="center",
                fontsize=7, color=DARK, zorder=4)

# ── Level 3: File references ──
file_y = card_y_start - 5 * card_gap - 0.1
for br in branches:
    box = FancyBboxPatch(
        (br["x"] - 2.0, file_y - 0.35), 4.0, 0.7,
        boxstyle="round,pad=0.05", facecolor="#ecf0f1", edgecolor=GRAY,
        linewidth=0.8, zorder=3
    )
    ax.add_patch(box)
    ax.text(br["x"], file_y, br["file"], ha="center", va="center",
            fontsize=6.5, color=DARK, family="monospace", zorder=4)

# ── Level 4: Usage status ──
status_y = file_y - 0.9
for br in branches:
    sc = GREEN if "ACTIVE" in br["status"] else RED
    box = FancyBboxPatch(
        (br["x"] - 2.0, status_y - 0.35), 4.0, 0.7,
        boxstyle="round,pad=0.05", facecolor=sc, edgecolor=DARK,
        linewidth=1.0, alpha=0.85, zorder=3
    )
    ax.add_patch(box)
    ax.text(br["x"], status_y + 0.08, br["status"], ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=4)
    ax.text(br["x"], status_y - 0.18, br["users"], ha="center", va="center",
            fontsize=6, color="white", zorder=4)

# ── Also show the vestigial modulate() ──
mod_x, mod_y = 16.5, 2.0
box = FancyBboxPatch(
    (mod_x - 2.0, mod_y - 0.45), 4.0, 0.9,
    boxstyle="round,pad=0.08", facecolor="#fadbd8", edgecolor=RED,
    linewidth=1.5, linestyle="--", zorder=3
)
ax.add_patch(box)
ax.text(mod_x, mod_y + 0.1, "modulate(x, shift, scale)",
        ha="center", va="center", fontsize=7, fontweight="bold", color=RED,
        family="monospace", zorder=4)
ax.text(mod_x, mod_y - 0.2, "DiT_modules/models.py L22\nDefined but NEVER called",
        ha="center", va="center", fontsize=6, color=RED, zorder=4)

# ── Formula boxes at bottom ──
formulas = [
    (3.5, r"$x' = \mathrm{LN}(x) \cdot (1 + s) + d$" + "\n" + r"$(s, d) = \mathrm{Linear}(\mathrm{SiLU}(t_\mathrm{emb}))$",
     BLUE),
    (10, r"$x' = \mathrm{RMS}(x) \cdot (1 + s) + d$" + "\n" + r"$(s, d, g) = \mathrm{Linear}(t_\mathrm{emb})$" + "\n" + r"$\mathrm{out} = \mathrm{residual} + x' \cdot g$",
     PURPLE),
    (16.5, r"$x' = x \cdot (1 + \gamma) + \beta$" + "\n" + r"$\gamma = \mathrm{MLP}(\bar{s})$, $\beta = \mathrm{MLP}(\bar{s})$",
     ORANGE),
]

for fx, formula, col in formulas:
    box = FancyBboxPatch(
        (fx - 2.2, 0.0), 4.4, 1.4,
        boxstyle="round,pad=0.1", facecolor="white", edgecolor=col,
        linewidth=2.0, zorder=3
    )
    ax.add_patch(box)
    ax.text(fx, 0.7, formula, ha="center", va="center",
            fontsize=7, color=DARK, zorder=4)

plt.tight_layout(pad=1.0)
plt.savefig("/home/luogang/SRC/Robot/starVLA/b/d/asset/adaln_family_tree.png",
            dpi=180, bbox_inches="tight")
print("Saved adaln_family_tree.png")
