"""
Classifier-Free Guidance (CFG) 引导效果可视化

展示条件分布、无条件分布和 CFG 引导后分布的关系，
以及不同引导强度 s 对生成结果的影响。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = ["DejaVu Sans", "WenQuanYi Micro Hei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def mvn_pdf(pos, mu, cov):
    d = pos - mu
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    exponent = -0.5 * np.einsum("...i,ij,...j", d, cov_inv, d)
    return np.exp(exponent) / (2 * np.pi * np.sqrt(det))

fig = plt.figure(figsize=(16, 10))

# ── Panel 1: 2D density comparison ──
ax1 = fig.add_subplot(1, 2, 1)

x = np.linspace(-4, 6, 300)
y = np.linspace(-4, 6, 300)
X, Y = np.meshgrid(x, y)
pos = np.stack([X, Y], axis=-1)

mu_uncond = np.array([1.0, 1.0])
cov_uncond = np.array([[2.0, 0.3], [0.3, 2.0]])
pdf_uncond = mvn_pdf(pos, mu_uncond, cov_uncond)

mu_cond = np.array([3.0, 3.0])
cov_cond = np.array([[0.8, 0.2], [0.2, 0.8]])
pdf_cond = mvn_pdf(pos, mu_cond, cov_cond)

cfg_scale = 2.5
log_guided = np.log(pdf_uncond + 1e-30) + cfg_scale * (
    np.log(pdf_cond + 1e-30) - np.log(pdf_uncond + 1e-30)
)
guided = np.exp(log_guided)
guided /= guided.sum() * (x[1] - x[0]) * (y[1] - y[0])

ax1.contour(X, Y, pdf_uncond, levels=5, colors="#7f8c8d", linestyles="dashed", alpha=0.6, linewidths=1.2)
ax1.contour(X, Y, pdf_cond, levels=5, colors="#2980b9", alpha=0.7, linewidths=1.5)
ax1.contourf(X, Y, guided, levels=15, cmap="Oranges", alpha=0.5)
ax1.contour(X, Y, guided, levels=5, colors="#e74c3c", linewidths=2.0)

ax1.plot(*mu_uncond, "s", color="#7f8c8d", markersize=10, label=r"$p_\theta(x_t | \varnothing)$ unconditional", zorder=5)
ax1.plot(*mu_cond, "o", color="#2980b9", markersize=10, label=r"$p_\theta(x_t | c)$ conditional", zorder=5)

mu_guided = mu_uncond + cfg_scale * (mu_cond - mu_uncond)
mu_guided = np.clip(mu_guided, -3, 5.5)
ax1.plot(*mu_guided, "^", color="#e74c3c", markersize=12, label=f"CFG guided ($s={cfg_scale}$)", zorder=5)

ax1.annotate("", xy=mu_guided, xytext=mu_uncond,
             arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5, connectionstyle="arc3,rad=0.1"))
ax1.annotate("", xy=mu_cond, xytext=mu_uncond,
             arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.5, connectionstyle="arc3,rad=-0.05"))

ax1.set_xlabel("Action dim 1", fontsize=12)
ax1.set_ylabel("Action dim 2", fontsize=12)
ax1.set_title("CFG: 2D Action Distribution Guidance", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax1.set_xlim(-3, 6)
ax1.set_ylim(-3, 6)
ax1.grid(True, alpha=0.2)

# ── Panel 2: 1D effect of guidance scale ──
ax2 = fig.add_subplot(1, 2, 2)

x1d = np.linspace(-4, 10, 500)
mu_u, sigma_u = 1.0, 1.8
mu_c, sigma_c = 3.5, 1.0

p_uncond = np.exp(-0.5 * ((x1d - mu_u) / sigma_u) ** 2) / (sigma_u * np.sqrt(2 * np.pi))
p_cond = np.exp(-0.5 * ((x1d - mu_c) / sigma_c) ** 2) / (sigma_c * np.sqrt(2 * np.pi))

ax2.fill_between(x1d, p_uncond, alpha=0.15, color="#7f8c8d")
ax2.plot(x1d, p_uncond, "--", color="#7f8c8d", lw=1.5, label=r"$p(x|\varnothing)$ uncond.")
ax2.plot(x1d, p_cond, "-", color="#2980b9", lw=2, label=r"$p(x|c)$ cond. ($s=1$)")

colors_s = ["#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
scales = [1.5, 2.5, 4.0, 7.0]

for s, col in zip(scales, colors_s):
    log_g = np.log(p_uncond + 1e-30) + s * (np.log(p_cond + 1e-30) - np.log(p_uncond + 1e-30))
    g = np.exp(log_g)
    g /= np.trapz(g, x1d)
    ax2.plot(x1d, g, "-", color=col, lw=1.8, label=f"$s = {s}$")

ax2.axvline(mu_c, color="#2980b9", ls=":", alpha=0.4)
ax2.axvline(mu_u, color="#7f8c8d", ls=":", alpha=0.4)

ax2.set_xlabel("Action value", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title("Effect of Guidance Scale $s$", fontsize=14, fontweight="bold")
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_xlim(-3, 9)
ax2.grid(True, alpha=0.2)

fig.tight_layout(pad=2.0)

# ── Add formula annotation at the bottom ──
fig.text(
    0.5, 0.01,
    r"$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s \cdot \left[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\right]$",
    ha="center", fontsize=14, style="italic",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", edgecolor="#bdc3c7")
)

plt.savefig("/home/luogang/SRC/Robot/starVLA/b/d/asset/cfg_guidance_viz.png", dpi=180, bbox_inches="tight")
print("Saved cfg_guidance_viz.png")
