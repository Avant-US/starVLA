"""
starVLA: 12 Fusion Mechanisms Comparison (融合机制对比)

4×3 grid of 12 mini-architecture diagrams, one per fusion mechanism.
Each cell shows a simplified block diagram of how modalities are fused.

Row 1: VLM Internal Fusion (VLM 内部融合)
Row 2: VLM→DiT Cross-Attention (VLM→DiT 交叉注意力)
Row 3: Dual Encoder / World Model (双编码器/世界模型)
Row 4: Specialized Mechanisms (特殊机制)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(28, 22))
ax.set_xlim(0, 30)
ax.set_ylim(0, 24)
ax.axis("off")

# ═══════════════ Color Palette ═══════════════
DARK = "#2c3e50"
GRAY = "#7f8c8d"
WHITE = "#ffffff"

# Row theme colors (background tint, accent / border)
ROW_BG = ["#dbeafe", "#d1fae5", "#fef3c7", "#ede9fe"]
ROW_ACCENT = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

# Component colors for mini-diagram boxes
C_VLM = "#3b82f6"  # blue  - VLM / LLM backbone
C_ACT = "#ef4444"  # red   - action output / heads
C_DIT = "#f59e0b"  # amber - DiT / diffusion blocks
C_ENC = "#10b981"  # green - vision encoders (DINO)
C_HEAD = "#8b5cf6"  # purple - output heads / hidden states
C_IN = "#14b8a6"  # teal  - input tokens / data
C_OP = "#6b7280"  # gray  - operations (concat, quantize, extract)
C_WM = "#06b6d4"  # cyan  - world model
C_GATE = "#ec4899"  # pink  - fusion / gate / regularizer
C_TEXT = "#d1d5db"  # light gray - text representation boxes

# ═══════════════ Grid Layout ═══════════════
CELL_W, CELL_H = 8.8, 4.5
COL_X = [5.5, 15.0, 24.5]
ROW_Y = [19.0, 13.8, 8.6, 3.4]
HDR_Y = [21.65, 16.45, 11.25, 6.05]


# ═══════════════ Drawing Helpers ═══════════════
def draw_box(x, y, w, h, text, color, fs=6.5, tc="white"):
    """Rounded box with centred label."""
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.06",
            facecolor=color,
            edgecolor=DARK,
            linewidth=1.0,
            alpha=0.95,
            zorder=5,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold",
        color=tc,
        zorder=6,
        linespacing=1.05,
    )


def draw_arrow(x1, y1, x2, y2, color=DARK, lw=1.3):
    """Straight arrow between two points."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=10,
            zorder=4,
        )
    )


def note(x, y, text, fs=4.5, color=GRAY):
    """Small italic annotation text."""
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center", style="italic", zorder=7)


# ═══════════════ Title ═══════════════
ax.text(
    15,
    23.2,
    "starVLA: 12 Fusion Mechanisms Comparison (融合机制对比)",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color=DARK,
)

# ═══════════════ Row Header Strips ═══════════════
for i, title in enumerate(
    [
        "Row 1 — VLM Internal Fusion  (VLM 内部融合)",
        "Row 2 — VLM→DiT Cross-Attention  (VLM→DiT 交叉注意力)",
        "Row 3 — Dual Encoder / World Model  (双编码器/世界模型)",
        "Row 4 — Specialized Mechanisms  (特殊机制)",
    ]
):
    ax.add_patch(
        FancyBboxPatch(
            (1.0, HDR_Y[i] - 0.22),
            28.0,
            0.44,
            boxstyle="round,pad=0.05",
            facecolor=ROW_ACCENT[i],
            edgecolor=ROW_ACCENT[i],
            linewidth=1.5,
            alpha=0.85,
            zorder=3,
        )
    )
    ax.text(15, HDR_Y[i], title, ha="center", va="center", fontsize=9, fontweight="bold", color=WHITE, zorder=4)

# ═══════════════ Cell Backgrounds ═══════════════
for r in range(4):
    for c in range(3):
        ax.add_patch(
            FancyBboxPatch(
                (COL_X[c] - CELL_W / 2, ROW_Y[r] - CELL_H / 2),
                CELL_W,
                CELL_H,
                boxstyle="round,pad=0.08",
                facecolor=ROW_BG[r],
                edgecolor=ROW_ACCENT[r],
                linewidth=1.2,
                alpha=0.45,
                zorder=2,
            )
        )


# ═══════════════ Cell Title & Label Helpers ═══════════════
def cell_title(row, col, num, text):
    cx = COL_X[col]
    ty = ROW_Y[row] + CELL_H / 2 - 0.3
    bw = 0.52 if num >= 10 else 0.42
    ax.add_patch(
        FancyBboxPatch(
            (cx - CELL_W / 2 + 0.12, ty - 0.17),
            bw,
            0.34,
            boxstyle="round,pad=0.04",
            facecolor=ROW_ACCENT[row],
            edgecolor=ROW_ACCENT[row],
            linewidth=1,
            zorder=7,
        )
    )
    ax.text(
        cx - CELL_W / 2 + 0.12 + bw / 2,
        ty,
        str(num),
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=WHITE,
        zorder=8,
    )
    ax.text(
        cx - CELL_W / 2 + 0.12 + bw + 0.12,
        ty,
        text,
        ha="left",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=DARK,
        zorder=8,
    )


def cell_label(row, col, text):
    cx = COL_X[col]
    by = ROW_Y[row] - CELL_H / 2 + 0.28
    ax.text(
        cx,
        by,
        text,
        ha="center",
        va="center",
        fontsize=5.5,
        color=GRAY,
        style="italic",
        zorder=8,
        bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor=GRAY, alpha=0.7, linewidth=0.5),
    )


# ══════════════════════════════════════════════════════════════
#  1.  Token Concatenation
# ══════════════════════════════════════════════════════════════
cell_title(0, 0, 1, "Token Concatenation")
cell_label(0, 0, "All Qwen-based frameworks")
x, y = COL_X[0], ROW_Y[0] - 0.1

draw_box(x - 3.0, y + 0.6, 1.5, 0.48, "Image\nTokens", C_IN, 6)
draw_box(x - 3.0, y - 0.4, 1.5, 0.48, "Text\nTokens", C_IN, 6)
draw_box(x - 1.1, y + 0.1, 0.65, 0.6, "⊕", C_OP, 10)
draw_box(x + 1.2, y + 0.1, 2.0, 0.55, "VLM\nSelf-Attention", C_VLM, 6.5)
draw_box(x + 3.5, y + 0.1, 1.3, 0.55, "Hidden\nStates", C_HEAD, 6)

draw_arrow(x - 2.25, y + 0.6, x - 1.43, y + 0.28)
draw_arrow(x - 2.25, y - 0.4, x - 1.43, y - 0.05)
draw_arrow(x - 0.78, y + 0.1, x + 0.2, y + 0.1)
draw_arrow(x + 2.2, y + 0.1, x + 2.85, y + 0.1)


# ══════════════════════════════════════════════════════════════
#  2.  In-sequence Action Token Regression
# ══════════════════════════════════════════════════════════════
cell_title(0, 1, 2, "Action Token Regression")
cell_label(0, 1, "QwenOFT")
x, y = COL_X[1], ROW_Y[0] - 0.1

draw_box(x - 3.1, y + 0.1, 1.9, 0.55, "Img ⊕ Txt ⊕\n[◆ Action]", C_IN, 5.5)
draw_box(x - 0.9, y + 0.1, 1.2, 0.55, "VLM", C_VLM, 7)
draw_box(x + 1.1, y + 0.1, 1.6, 0.55, "Extract\nat ◆ pos", C_OP, 6)
draw_box(x + 3.2, y + 0.1, 1.5, 0.55, "MLP\nL1 Head", C_ACT, 6)

draw_arrow(x - 2.15, y + 0.1, x - 1.5, y + 0.1)
draw_arrow(x - 0.3, y + 0.1, x + 0.3, y + 0.1)
draw_arrow(x + 1.9, y + 0.1, x + 2.45, y + 0.1)


# ══════════════════════════════════════════════════════════════
#  3.  Autoregressive Discrete Tokens
# ══════════════════════════════════════════════════════════════
cell_title(0, 2, 3, "Autoregressive Discrete")
cell_label(0, 2, "QwenFast")
x, y = COL_X[2], ROW_Y[0] - 0.1

draw_box(x - 3.0, y + 0.1, 1.5, 0.55, "Img ⊕ Txt", C_IN, 6.5)
draw_box(x - 1.1, y + 0.1, 1.2, 0.55, "VLM", C_VLM, 7)
draw_box(x + 1.0, y + 0.1, 1.8, 0.55, "robot_action\ntokens", C_OP, 5.5)
draw_box(x + 3.2, y + 0.1, 1.5, 0.55, "FAST\nDecode", C_ACT, 6.5)

draw_arrow(x - 2.25, y + 0.1, x - 1.7, y + 0.1)
draw_arrow(x - 0.5, y + 0.1, x + 0.1, y + 0.1)
draw_arrow(x + 1.9, y + 0.1, x + 2.45, y + 0.1)


# ══════════════════════════════════════════════════════════════
#  4.  Single-layer Cross-Attn DiT
# ══════════════════════════════════════════════════════════════
cell_title(1, 0, 4, "Single-layer Cross-Attn DiT")
cell_label(1, 0, "QwenGR00T, CosmosGR00T, MiniCPMGR00T, etc. (9 fw)")
x, y = COL_X[0], ROW_Y[1] - 0.05

# Top stream: VLM -> last_hidden
draw_box(x - 3.2, y + 0.65, 1.2, 0.48, "VLM", C_VLM, 7)
draw_box(x - 1.2, y + 0.65, 1.7, 0.48, "last_hidden\n(repeat N×)", C_OP, 5.5)
# Bottom stream: action/state input
draw_box(x - 2.5, y - 0.55, 2.2, 0.48, "Action + State\n+ Future", C_IN, 5.5)
# Cross-attention DiT
draw_box(x + 1.5, y + 0.05, 2.2, 0.80, "Cross-Attn\nDiT (×8)", C_DIT, 7)
# Output
draw_box(x + 3.6, y + 0.05, 0.85, 0.48, "Act", C_ACT, 6.5)

draw_arrow(x - 2.6, y + 0.65, x - 2.05, y + 0.65)
draw_arrow(x - 0.35, y + 0.65, x + 0.4, y + 0.33)
draw_arrow(x - 1.4, y - 0.55, x + 0.4, y - 0.13)
draw_arrow(x + 2.6, y + 0.05, x + 3.18, y + 0.05)

note(x + 0.1, y + 0.6, "K,V", 5, DARK)
note(x - 0.3, y - 0.48, "Q", 5, DARK)


# ══════════════════════════════════════════════════════════════
#  5.  Layer-wise Cross-Attn DiT
# ══════════════════════════════════════════════════════════════
cell_title(1, 1, 5, "Layer-wise Cross-Attn DiT")
cell_label(1, 1, "QwenPI/PI_v3, GemmaPI, etc. (6 frameworks)")
x, y = COL_X[1], ROW_Y[1] - 0.05

draw_box(x - 3.2, y + 0.1, 1.2, 0.55, "VLM", C_VLM, 7)
draw_box(x - 1.5, y + 0.1, 1.4, 0.55, "hidden\nstates[-N:]", C_OP, 5.5)
draw_box(x + 0.4, y + 0.1, 1.5, 0.55, "Linear proj\n(×N layers)", C_OP, 5.5)
draw_box(x + 2.5, y + 0.1, 1.6, 0.55, "Layer-wise\nDiT", C_DIT, 6)

draw_arrow(x - 2.6, y + 0.1, x - 2.2, y + 0.1)
draw_arrow(x - 0.8, y + 0.1, x - 0.35, y + 0.1)
draw_arrow(x + 1.15, y + 0.1, x + 1.7, y + 0.1)

note(x + 0.0, y - 0.4, "per-layer cross-attention coupling", 4.5)


# ══════════════════════════════════════════════════════════════
#  6.  Layer-wise QFormer Aggregation
# ══════════════════════════════════════════════════════════════
cell_title(1, 2, 6, "Layer-wise QFormer Aggregation")
cell_label(1, 2, "InternVLA-M1")
x, y = COL_X[2], ROW_Y[1] - 0.05

draw_box(x - 3.0, y + 0.55, 1.5, 0.48, "VLM\nper-layer", C_VLM, 6)
draw_box(x - 3.0, y - 0.4, 1.5, 0.48, "DINO\nfeatures", C_ENC, 6)
draw_box(x - 0.8, y + 0.08, 1.5, 0.6, "QFormer", C_GATE, 7)
draw_box(x + 1.3, y + 0.08, 1.2, 0.55, "DiT", C_DIT, 7)
draw_box(x + 3.0, y + 0.08, 1.0, 0.50, "Action", C_ACT, 6.5)

draw_arrow(x - 2.25, y + 0.55, x - 1.55, y + 0.25)
draw_arrow(x - 2.25, y - 0.4, x - 1.55, y - 0.08)
draw_arrow(x - 0.05, y + 0.08, x + 0.7, y + 0.08)
draw_arrow(x + 1.9, y + 0.08, x + 2.5, y + 0.08)


# ══════════════════════════════════════════════════════════════
#  7.  Dual Encoder Concatenation
# ══════════════════════════════════════════════════════════════
cell_title(2, 0, 7, "Dual Encoder Concatenation")
cell_label(2, 0, "QwenDual")
x, y = COL_X[0], ROW_Y[2] - 0.05

draw_box(x - 3.0, y + 0.55, 1.5, 0.48, "VLM\nfeatures", C_VLM, 6)
draw_box(x - 3.0, y - 0.4, 1.5, 0.48, "DINO\nfeatures", C_ENC, 6)
draw_box(x - 1.0, y + 0.08, 0.7, 0.6, "⊕", C_OP, 10)
draw_box(x + 1.2, y + 0.08, 1.8, 0.55, "DiT\ncross-attn", C_DIT, 6.5)
draw_box(x + 3.3, y + 0.08, 1.0, 0.50, "Action", C_ACT, 6.5)

draw_arrow(x - 2.25, y + 0.55, x - 1.35, y + 0.25)
draw_arrow(x - 2.25, y - 0.4, x - 1.35, y - 0.08)
draw_arrow(x - 0.65, y + 0.08, x + 0.3, y + 0.08)
draw_arrow(x + 2.1, y + 0.08, x + 2.8, y + 0.08)


# ══════════════════════════════════════════════════════════════
#  8.  World Model Feature Extraction
# ══════════════════════════════════════════════════════════════
cell_title(2, 1, 8, "World Model Feature Extraction")
cell_label(2, 1, "WanGR00T/PI/OFT, CosmoPredict2GR00T/PI/OFT")
x, y = COL_X[1], ROW_Y[2] - 0.05

draw_box(x - 3.2, y + 0.1, 1.4, 0.55, "WM DiT", C_WM, 7)
draw_box(x - 1.3, y + 0.1, 1.5, 0.55, "Intermediate\nFeatures", C_OP, 5.5)
draw_box(x + 0.7, y + 0.1, 1.4, 0.55, "Linear\nproj", C_OP, 6)
draw_box(x + 2.7, y + 0.1, 1.5, 0.55, "Action\nDiT", C_DIT, 6.5)

draw_arrow(x - 2.5, y + 0.1, x - 2.05, y + 0.1)
draw_arrow(x - 0.55, y + 0.1, x + 0.0, y + 0.1)
draw_arrow(x + 1.4, y + 0.1, x + 1.95, y + 0.1)

note(x + 2.7, y - 0.35, "cross-attn", 4.5)


# ══════════════════════════════════════════════════════════════
#  9.  Interleaved VLM + Action Expert
# ══════════════════════════════════════════════════════════════
cell_title(2, 2, 9, "Interleaved VLM + Action Expert")
cell_label(2, 2, "PI0, PI05")
x, y = COL_X[2], ROW_Y[2] - 0.05

draw_box(x - 3.2, y + 0.1, 1.5, 0.6, "Shared\nGemma", C_VLM, 6.5)
draw_box(x - 1.0, y + 0.1, 1.6, 0.55, "Joint\nQ/K/V Attn", C_GATE, 5.5)
draw_box(x + 1.3, y + 0.6, 1.3, 0.48, "VLM\nOutput", C_HEAD, 6)
draw_box(x + 1.3, y - 0.4, 1.3, 0.48, "Action\nExpert", C_ACT, 6)
draw_box(x + 3.2, y - 0.4, 1.2, 0.48, "Flow\nMatch", C_DIT, 6)

draw_arrow(x - 2.45, y + 0.1, x - 1.8, y + 0.1)
draw_arrow(x - 0.2, y + 0.25, x + 0.65, y + 0.6)
draw_arrow(x - 0.2, y - 0.05, x + 0.65, y - 0.4)
draw_arrow(x + 1.95, y - 0.4, x + 2.6, y - 0.4)


# ══════════════════════════════════════════════════════════════
# 10.  Discretized State Text Injection
# ══════════════════════════════════════════════════════════════
cell_title(3, 0, 10, "State Text Injection")
cell_label(3, 0, "QwenPI_v3, QwenOFT, PI05")
x, y = COL_X[0], ROW_Y[3] - 0.05

draw_box(x - 3.2, y + 0.1, 1.2, 0.55, "Robot\nState", C_IN, 6)
draw_box(x - 1.5, y + 0.1, 1.5, 0.55, "256-bin\nQuantize", C_OP, 6)
draw_box(x + 0.7, y + 0.1, 2.0, 0.55, "[STATE]\n95 133 ...", C_TEXT, 5.5, tc=DARK)
draw_box(x + 3.2, y + 0.1, 1.3, 0.55, "VLM\nInput", C_VLM, 6.5)

draw_arrow(x - 2.6, y + 0.1, x - 2.25, y + 0.1)
draw_arrow(x - 0.75, y + 0.1, x - 0.3, y + 0.1)
draw_arrow(x + 1.7, y + 0.1, x + 2.55, y + 0.1)

note(x + 2.2, y - 0.3, "inject as text", 4.5)


# ══════════════════════════════════════════════════════════════
# 11.  LangForce Dual-Branch
# ══════════════════════════════════════════════════════════════
cell_title(3, 1, 11, "LangForce Dual-Branch")
cell_label(3, 1, "LangForce")
x, y = COL_X[1], ROW_Y[3] - 0.05

draw_box(x - 3.0, y + 0.55, 1.6, 0.48, "Prior\n(V+A+L)", C_VLM, 6)
draw_box(x - 3.0, y - 0.4, 1.6, 0.48, "Posterior\n(V+L+A)", C_HEAD, 6)
draw_box(x - 0.5, y + 0.08, 1.8, 0.6, "LLR\nRegularizer", C_GATE, 6)
draw_box(x + 1.9, y + 0.08, 1.7, 0.55, "Hard Token\n/ Gate", C_ACT, 5.5)

draw_arrow(x - 2.2, y + 0.55, x - 1.4, y + 0.25)
draw_arrow(x - 2.2, y - 0.4, x - 1.4, y - 0.08)
draw_arrow(x + 0.4, y + 0.08, x + 1.05, y + 0.08)


# ══════════════════════════════════════════════════════════════
# 12.  VLA-Adapter Gated Multi-Source
# ══════════════════════════════════════════════════════════════
cell_title(3, 2, 12, "VLA-Adapter Gated Multi-Source")
cell_label(3, 2, "QwenAdapter")
x, y = COL_X[2], ROW_Y[3] - 0.05

draw_box(x - 3.2, y + 0.1, 1.5, 0.55, "action_query\ntokens", C_IN, 5.5)
draw_box(x - 1.2, y + 0.1, 1.4, 0.55, "Forward\nHook", C_OP, 6)
draw_box(x + 0.8, y + 0.1, 1.6, 0.55, "Multi-layer\nVLM feat", C_VLM, 5.5)
draw_box(x + 3.0, y + 0.1, 1.5, 0.55, "VLA-Adapter\nHead", C_HEAD, 5.5)

draw_arrow(x - 2.45, y + 0.1, x - 1.9, y + 0.1)
draw_arrow(x - 0.5, y + 0.1, x + 0.0, y + 0.1)
draw_arrow(x + 1.6, y + 0.1, x + 2.25, y + 0.1)


# ═══════════════ Save ═══════════════
plt.tight_layout(pad=1.0)
plt.savefig(
    "/home/luogang/SRC/Robot/starVLA/b/d/asset/fusion_mechanism_comparison.png",
    dpi=180,
    bbox_inches="tight",
)
print("Saved fusion_mechanism_comparison.png")
