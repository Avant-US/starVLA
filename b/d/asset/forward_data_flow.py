#!/usr/bin/env python3
"""
Forward Pass Data Flow -- 4 Representative starVLA Frameworks

Generates a 4-column diagram (one per framework) showing the
complete training-mode forward pass with tensor shapes and
key file references.

Frameworks:
  QwenGR00T  |  QwenPI_v3  |  QwenFast  |  WanGR00T
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Font / rendering setup ──
plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(28, 20))
ax.set_xlim(-0.5, 28.5)
ax.set_ylim(-0.5, 21)
ax.axis("off")

# ── Color palette ──
BLUE = "#2980b9"  # VLM backbone
GREEN = "#27ae60"  # Action head
ORANGE = "#e67e22"  # World model
PURPLE = "#8e44ad"  # Embeddings / projections
TEAL = "#16a085"  # Input / output
RED = "#c0392b"  # Loss
DARK = "#2c3e50"  # Headers / text
GRAY = "#7f8c8d"
LGRAY = "#ecf0f1"

# ── Layout constants ──
BW = 5.0  # box width
H = 0.85  # two-line box height
HS = 0.65  # single-line box height
STEP = 1.4  # vertical spacing

C1, C2, C3, C4 = 3.5, 10.5, 17.5, 24.5
Y = [17.5 - i * STEP for i in range(10)]
# Y = [17.5, 16.1, 14.7, 13.3, 11.9, 10.5, 9.1, 7.7, 6.3, 4.9]


# ── Drawing helpers ────────────────────────────────────────────


def draw_box(x, y, w, h, color, alpha=0.95, ls="-"):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor=DARK,
        linewidth=1.2,
        alpha=alpha,
        zorder=3,
        linestyle=ls,
    )
    ax.add_patch(p)


def nd(x, y, w, h, title, sub, color, tfs=8, sfs=6, tc="white", ls="-"):
    """Draw a node (rounded box with one or two text lines)."""
    draw_box(x, y, w, h, color, ls=ls)
    if sub:
        ax.text(
            x, y + h * 0.17, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=tc, zorder=4,
        )
        ax.text(
            x, y - h * 0.18, sub, ha="center", va="center",
            fontsize=sfs, color=tc, zorder=4, family="monospace", alpha=0.9,
        )
    else:
        ax.text(
            x, y, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=tc, zorder=4,
        )


def arr(x1, y1, x2, y2, color=DARK, lw=1.5):
    """Straight arrow."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
            lw=lw, mutation_scale=12, zorder=2,
        )
    )


def va(x, yf, hf, yt, ht, c=DARK):
    """Vertical arrow between two vertically-stacked boxes."""
    arr(x, yf - hf / 2, x, yt + ht / 2, c)


def ctx(x1, y1, x2, y2, label, rad=0.12, color=BLUE):
    """Curved dashed context arrow with a rotated label."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
            lw=1.2, mutation_scale=10, zorder=2,
            connectionstyle=f"arc3,rad={rad}", linestyle="--",
        )
    )
    lx = x1 + (-0.35 if rad > 0 else 0.35)
    ax.text(
        lx, (y1 + y2) / 2, label,
        ha="center", va="center", fontsize=5.5, color=color,
        style="italic", rotation=90, zorder=5,
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85),
    )


def fref(x, y, text):
    """File-reference label at the bottom of a column."""
    draw_box(x, y, BW, 0.5, LGRAY, alpha=0.7)
    ax.text(
        x, y, text, ha="center", va="center", fontsize=5.5,
        color=DARK, family="monospace", zorder=4,
    )


# ── Title + subtitle ─────────────────────────────────────────

ax.text(
    14, 20.5, "Forward Pass Data Flow — 4 starVLA Framework Variants",
    ha="center", va="center", fontsize=17, fontweight="bold", color=DARK,
)
ax.text(
    14, 20.0, "Training mode · top-to-bottom data flow with tensor shapes",
    ha="center", va="center", fontsize=10, color=GRAY, style="italic",
)

# ── Legend ────────────────────────────────────────────────────

leg_y = 19.5
for lx, ll, lc in [
    (4, "VLM Backbone", BLUE),
    (8, "Action Head", GREEN),
    (12, "World Model", ORANGE),
    (16, "Embed / Project", PURPLE),
    (20, "Input / Output", TEAL),
    (24, "Loss", RED),
]:
    draw_box(lx, leg_y, 2.8, 0.36, lc)
    ax.text(
        lx, leg_y, ll, ha="center", va="center", fontsize=6.5,
        fontweight="bold", color="white", zorder=4,
    )

# ── Column headers ───────────────────────────────────────────

hy = 18.7
for cx, h1, h2 in [
    (C1, "QwenGR00T", "VLM → Single-layer DiT"),
    (C2, "QwenPI_v3", "Layer-wise Cross-Attn + State Disc."),
    (C3, "QwenFast", "Autoregressive Discrete Tokens"),
    (C4, "WanGR00T", "World Model → Action Head"),
]:
    draw_box(cx, hy, 6.2, 0.8, DARK)
    ax.text(
        cx, hy + 0.13, h1, ha="center", va="center", fontsize=10,
        fontweight="bold", color="white", zorder=4,
    )
    ax.text(
        cx, hy - 0.16, h2, ha="center", va="center", fontsize=7,
        color="#bdc3c7", zorder=4,
    )

# Column separators (dotted vertical lines)
for sx in (7, 14, 21):
    ax.plot([sx, sx], [4.5, 18.2], color=LGRAY, lw=0.8, ls=":", alpha=0.5)


# ================================================================
# COL 1 -- QwenGR00T
# ================================================================
x = C1

nd(x, Y[0], BW, H, "Input", "[B,V,3,H,W]+str+[B,D_s]+[B,T,D_a]", TEAL)
va(x, Y[0], H, Y[1], H)

nd(x, Y[1], BW, H, "build_qwenvl_inputs()", "tokenize + image preprocess", BLUE)
va(x, Y[1], H, Y[2], H)

nd(x, Y[2], BW, H, "Qwen-VL Forward", "→ hidden_states[-1]  [B,L,1536]", BLUE)
va(x, Y[2], H, Y[3], H)

nd(x, Y[3], BW, H, "Repeat hidden states", "→ [B×T_chunk, L, D]", PURPLE)
va(x, Y[3], H, Y[4], H)

nd(
    x, Y[4], BW, H, "ActionEnc + StateMLP + FutureEmb",
    "[B,T,D] + [B,1,D] + [B,T,D]", PURPLE, tfs=7,
)
va(x, Y[4], H, Y[5], H)

nd(
    x, Y[5], BW, H, "Concat [state⊕future⊕action]",
    "→ [B, T+2, D]", PURPLE, tfs=7,
)
va(x, Y[5], H, Y[6], H)

nd(x, Y[6], BW, H, "Cross-Attention DiT (8 blks)", "← VLM hidden as context", GREEN)

# Context arrow: Repeat hidden states --> DiT (curved along left edge)
ctx(x - BW / 2, Y[3], x - BW / 2, Y[6], "context", rad=0.12, color=BLUE)

va(x, Y[6], H, Y[7], HS)
nd(x, Y[7], BW, HS, "Output: velocity pred [B,T,D_a]", None, TEAL, tfs=7)
va(x, Y[7], HS, Y[8], HS)
nd(x, Y[8], BW, HS, "Loss: MSE(v_pred, v_target)", None, RED, tfs=7)

fref(x, Y[8] - HS / 2 - 0.55, "QwenGR00T.py | GR00T_ActionHeader.py")


# ================================================================
# COL 2 -- QwenPI_v3
# ================================================================
x = C2

nd(x, Y[0], BW, H, "Input", "Imgs + Instr + State + Actions", TEAL)
va(x, Y[0], H, Y[1], H)

nd(x, Y[1], BW, H, "add_discretized_state()", "state→256 bins→append to text", PURPLE)
va(x, Y[1], H, Y[2], H)

nd(x, Y[2], BW, H, "build_qwenvl_inputs()", "tokenize + preprocess", BLUE)
va(x, Y[2], H, Y[3], H)

nd(x, Y[3], BW, H, "Qwen-VL Forward", "→ hidden_states[-N:] (N layers)", BLUE)
va(x, Y[3], H, Y[4], H)

nd(x, Y[4], BW, H, "Per-layer projection", "project_layers[i]→[B,L,D_proj]", PURPLE)
va(x, Y[4], H, Y[5], H)

nd(x, Y[5], BW, H, "LayerwiseFM ActionHead", "ActionEnc + layer-wise DiT", GREEN)

# Side annotation highlighting the per-layer correspondence
ax.text(
    x + BW / 2 + 0.1, (Y[4] + Y[5]) / 2,
    "each DiT block i\nuses VLM layer i",
    ha="left", va="center", fontsize=5, color=PURPLE, style="italic", zorder=4,
)

va(x, Y[5], H, Y[6], HS)
nd(x, Y[6], BW, HS, "Output: velocity pred [B,T,D_a]", None, TEAL, tfs=7)
va(x, Y[6], HS, Y[7], HS)
nd(x, Y[7], BW, HS, "Loss: MSE(v_pred, v_target)", None, RED, tfs=7)

fref(x, Y[7] - HS / 2 - 0.55, "QwenPI_v3.py | LayerwiseFM_ActionHeader.py")


# ================================================================
# COL 3 -- QwenFast
# ================================================================
x = C3

nd(x, Y[0], BW, H, "Input", "Imgs + Instr + Actions", TEAL)
va(x, Y[0], H, Y[1], H)

nd(x, Y[1], BW, H, "FAST Tokenizer", "[B,T,D_a]→[B,K] K≈50-200", PURPLE)
va(x, Y[1], H, Y[2], H)

nd(x, Y[2], BW, H, "Map → special tokens", "→ <robot_action_*> IDs", PURPLE)
va(x, Y[2], H, Y[3], H)

nd(x, Y[3], BW, H, "Build VLM input", "[Img]⊕[Text]⊕[action_tokens]", BLUE)
va(x, Y[3], H, Y[4], H)

nd(x, Y[4], BW, H, "Qwen-VL Forward", "labels masked before action tokens", BLUE)
va(x, Y[4], H, Y[5], HS)

nd(x, Y[5], BW, HS, "CE Loss (action token positions)", None, RED, tfs=7)

# ── Inference-only separator ──
sep_y = Y[5] - HS / 2 - 0.5
ax.plot(
    [x - BW / 2 + 0.3, x + BW / 2 - 0.3], [sep_y, sep_y],
    color=GRAY, lw=1.0, ls="--",
)
ax.text(
    x, sep_y + 0.18, "inference only",
    ha="center", va="center", fontsize=5.5, color=GRAY, style="italic",
)

inf_y = sep_y - 0.55
nd(x, inf_y, BW, H, "model.generate()", "→ decode tokens → [B,T,D_a]", GREEN, ls="--")

fref(x, inf_y - H / 2 - 0.55, "QwenFast.py")


# ================================================================
# COL 4 -- WanGR00T
# ================================================================
x = C4
pw = 2.2  # parallel-box width
lx, rx = x - 1.5, x + 1.5  # centres of the two parallel boxes

nd(x, Y[0], BW, H, "Input", "Imgs + Instr + State + Actions", TEAL)

# Fork: Input --> two parallel encoders
arr(lx, Y[0] - H / 2, lx, Y[1] + H / 2)
arr(rx, Y[0] - H / 2, rx, Y[1] + H / 2)

nd(lx, Y[1], pw, H, "UMT5 Text Enc", "[B,L_t,3072]", ORANGE, tfs=7, sfs=5.5)
nd(rx, Y[1], pw, H, "Wan VAE Enc", "[B,C,T,h,w]", ORANGE, tfs=7, sfs=5.5)

# Merge: parallel encoders --> WanTransformer
arr(lx, Y[1] - H / 2, x, Y[2] + H / 2)
arr(rx, Y[1] - H / 2, x, Y[2] + H / 2)

nd(x, Y[2], BW, H, "WanTransformer3D DiT", "hook→features [B,L_wm,3072]", ORANGE)
va(x, Y[2], H, Y[3], H)

nd(x, Y[3], BW, H, "wm_projector", "Linear(3072, D_action)", PURPLE)
va(x, Y[3], H, Y[4], H)

nd(x, Y[4], BW, H, "ActionEncoder", "actions+timestep→[B,T,D]", GREEN)
va(x, Y[4], H, Y[5], H)

nd(x, Y[5], BW, H, "Cross-Attention DiT", "← WM features as context", GREEN)

# Context arrow: wm_projector --> Cross-Attention DiT (curved along right edge)
ctx(x + BW / 2, Y[3], x + BW / 2, Y[5], "WM ctx", rad=-0.2, color=ORANGE)

va(x, Y[5], H, Y[6], HS)
nd(x, Y[6], BW, HS, "Output: velocity pred [B,T,D_a]", None, TEAL, tfs=7)
va(x, Y[6], HS, Y[7], HS)
nd(x, Y[7], BW, HS, "Loss: MSE(v_pred, v_target)", None, RED, tfs=7)

fref(x, Y[7] - HS / 2 - 0.55, "WanGR00T.py | Wan2.py")


# ── Save ──────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
plt.savefig(
    "/home/luogang/SRC/Robot/starVLA/b/d/asset/forward_data_flow.png",
    dpi=180,
    bbox_inches="tight",
)
print("Saved forward_data_flow.png")
