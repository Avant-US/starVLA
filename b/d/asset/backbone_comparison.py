"""
starVLA Perception Backbone Comparison Diagram (感知骨干对比)

Generates a grouped tree/diagram showing all perception backbones:
  Section 1: VLM Family (Vision-Language Models) — Blue theme
  Section 2: World Model Family — Orange theme
  Section 3: Supplementary Encoders — Green theme
  Section 4: Summary comparison table with parameter scale bars
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(24, 18))
ax.set_xlim(-1, 25)
ax.set_ylim(-2.5, 19)
ax.axis("off")

# ── Color constants ──
BLUE = "#2980b9"
GREEN = "#27ae60"
ORANGE = "#e67e22"
GRAY = "#7f8c8d"
DARK = "#2c3e50"
WHITE = "#ffffff"
LIGHT_BLUE = "#d6eaf8"
LIGHT_GREEN = "#d5f5e3"
LIGHT_ORANGE = "#fdebd0"
DEEP_BLUE = "#1a5276"
DEEP_ORANGE = "#a04000"
DEEP_GREEN = "#1e8449"
BG_GRAY = "#ecf0f1"


def draw_box(x, y, w, h, label, color, fontsize=9, textcolor="white"):
    """Draw a rounded box with centered bold label (main node style)."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=DARK,
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)


def draw_detail(x, y, w, h, label, bg, tc=None, fs=7, edge=None, family=None, bold=False):
    """Draw a detail / info card box."""
    if tc is None:
        tc = DARK
    if edge is None:
        edge = GRAY
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=bg,
        edgecolor=edge,
        linewidth=0.8,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(box)
    kw = dict(ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=4)
    if family:
        kw["family"] = family
    ax.text(x, y, label, **kw)


def draw_arrow(x1, y1, x2, y2, color=DARK, lw=2.0):
    """Draw a fancy arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", color=color, lw=lw, mutation_scale=15, zorder=2
    )
    ax.add_patch(arrow)


def section_bg(x, y, w, h, color):
    """Draw subtle dashed background rectangle for section grouping."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.2",
        facecolor=color,
        edgecolor=color,
        linewidth=1.5,
        alpha=0.05,
        zorder=0,
        linestyle="--",
    )
    ax.add_patch(box)


# ═══════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════
ax.text(
    12, 18.5, "starVLA Perception Backbone Comparison (感知骨干对比)", ha="center", va="center",
    fontsize=20, fontweight="bold", color=DARK,
)
ax.text(
    12, 18.0, "All perception backbone architectures organized by family", ha="center", va="center",
    fontsize=12, color=GRAY, style="italic",
)

# ═══════════════════════════════════════════════════════════
# Section 1: VLM Family (Blue theme)   y: 17.3 → 11.1
# ═══════════════════════════════════════════════════════════
section_bg(-0.5, 11.1, 25.0, 6.4, BLUE)
ax.text(0.3, 17.3, "Section 1", ha="left", va="center", fontsize=9, color=BLUE, style="italic", alpha=0.6)

# Section header box
draw_box(12, 17.0, 10, 0.65, "VLM Family  (Vision-Language Models)", BLUE, 12)

# Column width for VLM cards
CW = 4.8

vlm_cols = [
    dict(
        name="Qwen-VL Series", x=3,
        arch="ViT patch14\nhidden = 1536 / 3584\nNaViT + mrope",
        variants="Qwen2.5-VL\nQwen3-VL (improved)\nQwen3.5-VL (latest)",
        used="Used by:\nQwenGR00T, QwenPI, QwenFast,\nQwenOFT, QwenDual,\nQwenAdapter, QwenDiscreteDiffusion",
        file="vlm/QWen*.py",
    ),
    dict(
        name="PaliGemma\n(SigLIP + Gemma)", x=9,
        arch="SigLIP ViT patch14\nhidden = 1152",
        variants="Gemma 2B / 7B\nLLM decoder",
        used="Used by:\nPI0, PI05",
        file="vlm/PaliGemma.py",
    ),
    dict(
        name="MiniCPM-V", x=15,
        arch="OmniLMM\nvisual encoder",
        variants="MiniCPM LLM",
        used="Used by:\nMiniCPMGR00T, MiniCPMPI",
        file="vlm/MiniCPMV.py",
    ),
    dict(
        name="Gemma4", x=21,
        arch="Built-in\nvision encoder",
        variants="",
        used="Used by:\nGemma4GR00T, Gemma4PI",
        file="vlm/Gemma4.py",
    ),
]

# Fixed y-positions for each detail row
NY = 15.8   # name
AY = 14.7   # architecture
VY = 13.7   # variants
UY = 12.65  # used-by
FY = 11.75  # file ref

for c in vlm_cols:
    x = c["x"]

    # Arrow from header to name box
    draw_arrow(12, 16.67, x, 16.15, color=BLUE, lw=2)

    # Name box
    draw_box(x, NY, CW, 0.7, c["name"], BLUE, 9)

    # Architecture details
    draw_detail(x, AY, CW, 0.95, c["arch"], LIGHT_BLUE, fs=7.5)

    # Model variants
    txt = c["variants"] if c["variants"] else "(integrated)"
    tc = DARK if c["variants"] else GRAY
    draw_detail(x, VY, CW, 0.85, txt, "#aed6f1", tc=tc, fs=7)

    # Used-by box
    draw_detail(x, UY, CW, 1.1, c["used"], "#85c1e9", tc=DEEP_BLUE, edge=BLUE, bold=True, fs=6.5)

    # File reference
    draw_detail(x, FY, CW, 0.45, "starVLA/model/modules/\n" + c["file"], BG_GRAY, fs=5.5, family="monospace")


# ═══════════════════════════════════════════════════════════
# Section 2: World Model Family (Orange theme)   y: 10.5 → 4.5
# ═══════════════════════════════════════════════════════════
section_bg(-0.5, 4.5, 25.0, 6.0, ORANGE)
ax.text(0.3, 10.3, "Section 2", ha="left", va="center", fontsize=9, color=ORANGE, style="italic", alpha=0.6)

draw_box(12, 10.0, 7, 0.65, "World Model Family", ORANGE, 12)

WCW = 7.0  # wider columns for world-model cards

wm_cols = [
    dict(
        name="CosmoPredict2", x=6,
        comp=[
            ("T5 Text Encoder", "hidden = 2048"),
            ("CosmoPredict2 VAE", "48-ch latent, 8× spatial compression"),
            ("CosmosTransformer3DModel DiT", ""),
        ],
        used="Used by: CosmoPredict2GR00T,\nCosmoPredict2PI, CosmoPredict2OFT",
        file="world_model/CosmoPredict2.py",
    ),
    dict(
        name="Wan2.2-TI2V", x=18,
        comp=[
            ("UMT5 Text Encoder", "hidden = 3072"),
            ("AutoencoderKLWan VAE", "48-ch latent"),
            ("WanTransformer3D DiT", "hidden = 3072, heads = 48"),
        ],
        used="Used by: WanGR00T,\nWanPI, WanOFT",
        file="world_model/Wan2.py",
    ),
]

WNY = 8.8                      # name
WCY = [7.95, 7.2, 6.45]        # three component rows
WUY = 5.65                     # used-by
WFY = 5.05                     # file ref

for c in wm_cols:
    x = c["x"]

    draw_arrow(12, 9.67, x, 9.15, color=ORANGE, lw=2)
    draw_box(x, WNY, WCW, 0.65, c["name"], ORANGE, 10)

    for i, (cn, cd) in enumerate(c["comp"]):
        lbl = cn + ("\n" + cd if cd else "")
        draw_detail(x, WCY[i], WCW, 0.6, lbl, LIGHT_ORANGE, fs=7)

    draw_detail(x, WUY, WCW, 0.65, c["used"], "#f5cba7", tc=DEEP_ORANGE, edge=ORANGE, bold=True, fs=7)
    draw_detail(x, WFY, WCW, 0.4, "starVLA/model/modules/" + c["file"], BG_GRAY, fs=5.5, family="monospace")


# ═══════════════════════════════════════════════════════════
# Section 3: Supplementary Encoders (Green theme)   y: 4.4 → 2.0
# ═══════════════════════════════════════════════════════════
section_bg(-0.5, 2.0, 25.0, 2.4, GREEN)
ax.text(0.3, 4.2, "Section 3", ha="left", va="center", fontsize=9, color=GREEN, style="italic", alpha=0.6)

draw_box(5, 4.0, 6, 0.55, "Supplementary Encoders", GREEN, 11)

# DINOv2 main box
draw_box(5, 3.3, 5.5, 0.55, "DINOv2 (Self-supervised ViT)", GREEN, 9)
draw_arrow(5, 3.72, 5, 3.59, color=GREEN, lw=1.5)

# Four variant boxes
for i, (vn, vd) in enumerate([
    ("ViT-S/14", "384d"),
    ("ViT-B/14", "768d"),
    ("ViT-L/14", "1024d"),
    ("ViT-G/14", "1408d"),
]):
    vx = 1.5 + i * 2.5
    draw_detail(vx, 2.65, 2.2, 0.4, f"{vn} ({vd})", LIGHT_GREEN, fs=6.5)
    draw_arrow(5, 3.02, vx, 2.87, color=GREEN, lw=1.0)

# Input pipeline
draw_detail(
    5, 2.15, 9.5, 0.3,
    "Input: Resize(224) → ToTensor → ImageNet normalize",
    "#eafaf1", fs=6.5, family="monospace",
)

# Used-by (right side, connected by arrow)
draw_detail(
    17, 3.55, 8, 0.55,
    "Used by: InternVLA-M1, QwenDual (supplementary spatial features)",
    "#a9dfbf", tc=DEEP_GREEN, edge=GREEN, bold=True, fs=7,
)
draw_arrow(7.8, 3.3, 13.0, 3.55, color=GREEN, lw=1.5)

# File reference (right side)
draw_detail(17, 2.9, 8, 0.35, "starVLA/model/modules/dino_model/dino.py", BG_GRAY, fs=6, family="monospace")


# ═══════════════════════════════════════════════════════════
# Section 4: Summary Comparison Table   y: 1.6 → −1.8
# ═══════════════════════════════════════════════════════════
ax.text(12, 1.55, "Summary Comparison", ha="center", va="center", fontsize=14, fontweight="bold", color=DARK)

# Table geometry
headers = ["Backbone", "Hidden Dim", "~Params", "Type", "Key Frameworks"]
cw_t = [3.2, 2.8, 2.0, 1.8, 6.5]       # column widths
tl = 0.5                                 # table left edge
ccx = []                                 # column centers
cx = tl
for w in cw_t:
    ccx.append(cx + w / 2)
    cx += w
bx0 = cx + 0.5                          # bar column left edge
bw_max = 5.0                            # max bar width
rh = 0.38                               # row height
hy = 1.1                                # header row y

# Header row
for j, h in enumerate(headers):
    box = FancyBboxPatch(
        (ccx[j] - cw_t[j] / 2, hy - rh / 2), cw_t[j], rh,
        boxstyle="round,pad=0.02", facecolor=DARK, edgecolor=DARK, linewidth=0.8, zorder=3,
    )
    ax.add_patch(box)
    ax.text(ccx[j], hy, h, ha="center", va="center", fontsize=7.5, fontweight="bold", color=WHITE, zorder=4)

ax.text(bx0 + bw_max / 2, hy, "Param Scale", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=DARK)

# Data rows: (name, hidden_dim, params, type, frameworks, bg_color, bar_fraction, bar_color)
table_rows = [
    ("Qwen2.5-VL",   "1536 / 3584", "7B+",      "VLM", "QwenGR00T, QwenPI, QwenFast, QwenOFT...", LIGHT_BLUE,   0.50, BLUE),
    ("SigLIP+Gemma",  "1152 + 2048", "2–7B", "VLM", "PI0, PI05",                               LIGHT_BLUE,   0.35, BLUE),
    ("MiniCPM-V",     "—",      "3B",        "VLM", "MiniCPMGR00T, MiniCPMPI",                  LIGHT_BLUE,   0.22, BLUE),
    ("Gemma4",        "—",      "4B+",       "VLM", "Gemma4GR00T, Gemma4PI",                    LIGHT_BLUE,   0.30, BLUE),
    ("CosmoPredict2", "2048",        "14B",       "WM",  "CosmoPredict2GR00T / PI / OFT",           LIGHT_ORANGE,  1.00, ORANGE),
    ("Wan2.2-TI2V",   "3072",        "14B",       "WM",  "WanGR00T, WanPI, WanOFT",                 LIGHT_ORANGE,  1.00, ORANGE),
    ("DINOv2",        "384–1408", "22M–1.1B", "SSL", "InternVLA-M1, QwenDual",             LIGHT_GREEN,   0.08, GREEN),
]

for i, (nm, hd, pr, tp, fw, bg, bf, bc) in enumerate(table_rows):
    ry = hy - (i + 1) * rh
    for j, cell in enumerate([nm, hd, pr, tp, fw]):
        box = FancyBboxPatch(
            (ccx[j] - cw_t[j] / 2, ry - rh / 2), cw_t[j], rh,
            boxstyle="round,pad=0.02", facecolor=bg, edgecolor=GRAY, linewidth=0.5, zorder=3,
        )
        ax.add_patch(box)
        ax.text(ccx[j], ry, cell, ha="center", va="center", fontsize=6.5, color=DARK, zorder=4)

    # Parameter-scale bar
    bw = max(bf * bw_max, 0.15)
    bar = FancyBboxPatch(
        (bx0, ry - rh / 4), bw, rh / 2,
        boxstyle="round,pad=0.02", facecolor=bc, edgecolor=bc, linewidth=0.5, alpha=0.6, zorder=3,
    )
    ax.add_patch(bar)


# ── Legend ──
for i, (lc, lt) in enumerate([(BLUE, "VLM"), (ORANGE, "World Model"), (GREEN, "Supplementary")]):
    lx = 16.5 + i * 2.8
    box = FancyBboxPatch(
        (lx, -2.2), 0.5, 0.25,
        boxstyle="round,pad=0.02", facecolor=lc, edgecolor=lc, linewidth=0.5, zorder=3,
    )
    ax.add_patch(box)
    ax.text(lx + 0.7, -2.08, lt, ha="left", va="center", fontsize=7, color=DARK, zorder=4)


plt.tight_layout(pad=1.0)
plt.savefig(
    "/home/luogang/SRC/Robot/starVLA/b/d/asset/backbone_comparison.png",
    dpi=180, bbox_inches="tight",
)
print("Saved backbone_comparison.png")
