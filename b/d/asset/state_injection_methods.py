"""
State Encoding Methods in starVLA (状态编码方式对比)

3-panel comparison of Continuous MLP Encoding, Discretized Text Injection,
and ProprioProjector state encoding methods with complete data flows.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(26, 14))
ax.set_xlim(-0.5, 26.5)
ax.set_ylim(-0.8, 15)
ax.axis("off")

# ── Color palette ──
BLUE = "#2980b9"
BLUE_LIGHT = "#d6eaf8"
PURPLE = "#8e44ad"
PURPLE_LIGHT = "#e8daef"
GREEN = "#27ae60"
GREEN_LIGHT = "#d5f5e3"
DARK = "#2c3e50"
GRAY = "#7f8c8d"
RED = "#c0392b"
TEAL = "#16a085"
LIGHT_GRAY = "#ecf0f1"

# Panel centres and box dimensions
PX = [4.3, 13.0, 21.7]
BOX_W = 7.0
BOX_H = 0.48
IO_H = 0.56
DEST_H = 0.52


# ═══════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════


def draw_box(x, y, w, h, label, fc, tc="white", fs=8, ec=None, lw=1.5, ls="-", alpha=1.0, family=None, fw="bold"):
    """Draw a rounded rectangle with centred text."""
    ec = ec or DARK
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        linestyle=ls,
        zorder=3,
    )
    ax.add_patch(patch)
    kw = dict(ha="center", va="center", fontsize=fs, fontweight=fw, color=tc, zorder=4)
    if family:
        kw["family"] = family
    ax.text(x, y, label, **kw)


def draw_arrow(x1, y1, x2, y2, color=DARK, lw=2.0):
    """Draw a directed arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        color=color,
        lw=lw,
        mutation_scale=14,
        zorder=2,
    )
    ax.add_patch(arrow)


def _hh(stype):
    """Half-height of a box by step type."""
    return {"input": IO_H, "output": IO_H, "dest": DEST_H}.get(stype, BOX_H) / 2


def draw_flow(px, steps, col, light, y_top=11.8, y_bot=7.0):
    """Draw a vertical flow diagram for one panel."""
    n = len(steps)
    dy = (y_top - y_bot) / (n - 1) if n > 1 else 0

    for i, (st, lbl) in enumerate(steps):
        y = y_top - i * dy

        if st == "input":
            draw_box(px, y, BOX_W, IO_H, lbl, light, tc=DARK, fs=7.5, ec=col, lw=2.0)
        elif st == "output":
            draw_box(px, y, BOX_W, IO_H, lbl, col, tc="white", fs=7.5, lw=2.0)
        elif st == "dest":
            draw_box(px, y, BOX_W, DEST_H, lbl, "white", tc=col, ec=col, lw=2.0, ls="--", fs=7.5)
        else:
            draw_box(px, y, BOX_W, BOX_H, lbl, col, fs=7.5)

        # Arrow from the previous step
        if i > 0:
            yp = y_top - (i - 1) * dy
            draw_arrow(px, yp - _hh(steps[i - 1][0]) - 0.02, px, y + _hh(st) + 0.02, color=col)


# ═══════════════════════════════════════════════════════
#  Title
# ═══════════════════════════════════════════════════════

ax.text(
    13, 14.5, "State Encoding Methods in starVLA", ha="center", va="center", fontsize=20, fontweight="bold", color=DARK
)
ax.text(13, 14.0, "状态编码方式对比", ha="center", va="center", fontsize=14, color=GRAY)


# ═══════════════════════════════════════════════════════
#  Panel backgrounds & titles
# ═══════════════════════════════════════════════════════

panel_meta = [
    ("A: Continuous MLP Encoding", "连续 MLP 编码", BLUE, BLUE_LIGHT),
    ("B: Discretized Text Injection", "离散化文本注入", PURPLE, PURPLE_LIGHT),
    ("C: ProprioProjector", "本体感知投影器", GREEN, GREEN_LIGHT),
]

for px, (en, cn, col, lt) in zip(PX, panel_meta):
    bg = FancyBboxPatch(
        (px - BOX_W / 2 - 0.6, 3.3),
        BOX_W + 1.2,
        10.1,
        boxstyle="round,pad=0.15",
        facecolor=lt,
        edgecolor=col,
        linewidth=1.5,
        alpha=0.2,
        zorder=1,
    )
    ax.add_patch(bg)
    ax.text(px, 13.0, en, ha="center", va="center", fontsize=11, fontweight="bold", color=col)
    ax.text(px, 12.55, cn, ha="center", va="center", fontsize=9, color=col)


# ═══════════════════════════════════════════════════════
#  Panel A – Continuous MLP Encoding (Blue)
# ═══════════════════════════════════════════════════════

draw_flow(
    PX[0],
    [
        ("input", r"Raw State Vector  $s \in \mathbb{R}^{D_s}$" "\n(joint angles, EEF pos/rot, gripper)"),
        ("flow", r"Linear($D_s \rightarrow D_{hidden}$)"),
        ("flow", "ReLU Activation"),
        ("flow", r"Linear($D_{hidden} \rightarrow D_{model}$)"),
        ("output", r"State Embedding  $s_{emb} \in \mathbb{R}^{D_{model}}$"),
        ("dest", r"$\rightarrow$ DiT cross-attention" "\n(concat along sequence dim)"),
    ],
    BLUE,
    BLUE_LIGHT,
)


# ═══════════════════════════════════════════════════════
#  Panel B – Discretized Text Injection (Purple)
# ═══════════════════════════════════════════════════════

draw_flow(
    PX[1],
    [
        ("input", r"Raw State Vector  $s \in \mathbb{R}^{D_s}$"),
        ("flow", "Per-dimension normalize to [0, 1]"),
        ("flow", r"Quantize: $bin_{id} = \mathrm{round}(s_{norm} \times 255)$"),
        ("flow", 'Convert to text: "95 133 203 44 127 88 201"'),
        ("flow", 'Wrap: instruction + " [STATE] ... [ACTION]"'),
        ("flow", "Standard VLM tokenization → token IDs"),
        ("output", "Part of VLM text sequence"),
        ("dest", r"$\rightarrow$ VLM text understanding pathway"),
    ],
    PURPLE,
    PURPLE_LIGHT,
)


# ═══════════════════════════════════════════════════════
#  Panel C – ProprioProjector (Green)
# ═══════════════════════════════════════════════════════

draw_flow(
    PX[2],
    [
        ("input", r"Raw State Vector  $s \in \mathbb{R}^{D_s}$"),
        ("flow", r"Linear($D_s \rightarrow D_{inner}$)"),
        ("flow", "GELU Activation"),
        ("flow", r"Linear($D_{inner} \rightarrow D_{llm\_hidden}$)"),
        ("output", r"Proprio Embedding  $\in \mathbb{R}^{D_{llm\_hidden}}$"),
        ("dest", r"$\rightarrow$ VLM sequence as special tokens" "\n(injected via forward hook)"),
    ],
    GREEN,
    GREEN_LIGHT,
)


# ═══════════════════════════════════════════════════════
#  Formula boxes (below each flow diagram)
# ═══════════════════════════════════════════════════════

FORM_Y = 6.0

for px, formula, col in [
    (PX[0], r"$s_{emb} = W_2 \cdot \mathrm{ReLU}(W_1 \cdot s + b_1) + b_2$", BLUE),
    (
        PX[1],
        r"$b_i = \lfloor \frac{s_i - s_{min}}{s_{max} - s_{min}} \times 255 + 0.5 \rfloor$",
        PURPLE,
    ),
    (PX[2], r"$p = W_2 \cdot \mathrm{GELU}(W_1 \cdot s + b_1) + b_2$", GREEN),
]:
    draw_box(px, FORM_Y, BOX_W, 0.65, formula, "white", tc=DARK, ec=col, lw=2.0, fs=8, fw="normal")


# ═══════════════════════════════════════════════════════
#  File references, usage info, advantages / disadvantages
# ═══════════════════════════════════════════════════════

panel_info = [
    (
        "GR00T_ActionHeader.py (state_encoder = MLP)",
        "Used by: QwenGR00T, QwenPI_v3, WanGR00T, CosmosGR00T, ...",
        "+ Preserves continuous precision",
        "- Separate encoder needed, not VLM-native",
        BLUE,
    ),
    (
        "QwenPI_v3.py: add_discretized_state_to_instruction()",
        "Used by: QwenPI_v3, QwenOFT (optional), PI05",
        "+ No extra params, leverages VLM text understanding",
        "- Quantization error, longer sequence length",
        PURPLE,
    ),
    (
        "QwenAdapter.py: ProprioProjector",
        "Used by: QwenAdapter",
        "+ Projects directly to LLM embedding space",
        "- Fixed architecture, single-vector state only",
        GREEN,
    ),
]

for px, (fi, us, ad, di, col) in zip(PX, panel_info):
    ax.text(px, 5.1, fi, ha="center", va="center", fontsize=6.5, color=DARK, family="monospace", style="italic")
    ax.text(px, 4.6, us, ha="center", va="center", fontsize=7, fontweight="bold", color=col)
    ax.text(px, 4.1, ad, ha="center", va="center", fontsize=7, color=TEAL)
    ax.text(px, 3.7, di, ha="center", va="center", fontsize=7, color=RED)


# ═══════════════════════════════════════════════════════
#  Comparison table
# ═══════════════════════════════════════════════════════

ax.text(13, 2.7, "Comparison Table  (对比表)", ha="center", va="center", fontsize=12, fontweight="bold", color=DARK)

COL_W = 5.0
x0 = 0.5
ccx = [x0 + COL_W * (i + 0.5) for i in range(5)]
ROW_H = 0.55
HDR_Y = 2.05

headers = [
    "Method\n方法",
    "Parameters\n参数量",
    "Quantization Error\n量化误差",
    "VLM Integration\nVLM 集成方式",
    "Flexibility\n灵活性",
]

rows = [
    (
        ["MLP Encoding\n(连续 MLP)", r"$\sim D_s \times D_{model}$", "None", "External (DiT)", "High"],
        BLUE,
    ),
    (
        ["Discretization\n(离散化文本)", "0", "~0.4% (1/256)", "Native (text)", "Medium"],
        PURPLE,
    ),
    (
        ["ProprioProjector\n(本体投影)", r"$\sim 2 D_s \times D_{inner}$", "None", "Native (embedding)", "Medium"],
        GREEN,
    ),
]

# Header row
for cx, h in zip(ccx, headers):
    draw_box(cx, HDR_Y, COL_W - 0.12, ROW_H, h, DARK, tc="white", fs=7, fw="bold")

# Data rows
for r, (cells, rc) in enumerate(rows):
    ry = HDR_Y - (r + 1) * (ROW_H + 0.06)
    for c, (cx, cell) in enumerate(zip(ccx, cells)):
        fc = rc if c == 0 else LIGHT_GRAY
        tc = "white" if c == 0 else DARK
        fw = "bold" if c == 0 else "normal"
        draw_box(cx, ry, COL_W - 0.12, ROW_H, cell, fc, tc=tc, fs=7, fw=fw, ec=GRAY, lw=0.8)


# ═══════════════════════════════════════════════════════
#  Save
# ═══════════════════════════════════════════════════════

plt.tight_layout(pad=1.0)
plt.savefig(
    "/home/luogang/SRC/Robot/starVLA/b/d/asset/state_injection_methods.png",
    dpi=180,
    bbox_inches="tight",
)
print("Saved state_injection_methods.png")
