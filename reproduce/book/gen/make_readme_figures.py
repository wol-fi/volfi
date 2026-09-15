#!/usr/bin/env python3
"""make_readme_figures.py -- the two README figures of v0.3.0, from the checked-in results.

  docs/figures/cpu_one_binary.png   ns/quote on the market feed, all methods in one binary, per ISA
                                    (reproduce/book/results/cpu_all_20260914_185859_clean.txt)
  docs/figures/gpu_book_kernel.png  ns/quote on one H100 PCIe, kernel-resident and with transfers
                                    (results/gpu_near_run_2026-09-14*.txt, pde_gpu_run_2026-09-14*.txt,
                                    reproduce/results/gpu_run_2026-07-27.txt for the routed book)

Run from anywhere:  python3 reproduce/book/gen/make_readme_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[3] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# blue shades = machine precision on the feed, red = not (the PDE table method); same as the paper
C_LBR, C_PDE, C_ROUTED, C_BOOK = "#1b4f9c", "#d62728", "#4f93e0", "#9ec7f2"

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2); ax.xaxis.label.set_color(INK2)

# ---------------------------------------------------------------- CPU, one binary
# cpu_all_20260914_185859_clean.txt, full feed, no LTO: LBR | PDE scalar | routed batch | book batch
builds = ["AVX-512", "AVX2", "no SIMD"]
series = [
    ("Let's Be Rational (scalar, release flags)", C_LBR,    [194, 193, 220]),
    ("PDE table method (scalar)",                 C_PDE,    [76,  86,  89]),
    ("routed charts, batch (v0.2)",               C_ROUTED, [45,  89,  578]),
    ("book kernel, batch (v0.3)",                 C_BOOK,   [29,  51,  352]),
]
fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=200)
n, k = len(builds), len(series); w = 0.19; gap = 0.02
for j, (name, col, vals) in enumerate(series):
    xs = [i + (j - (k - 1) / 2) * (w + gap) for i in range(n)]
    ax.bar(xs, vals, width=w, color=col, label=name, zorder=3)
    for x, v in zip(xs, vals):
        ax.text(x, v + 8, str(v), ha="center", va="bottom", fontsize=8, color=INK)
ax.set_xticks(range(n)); ax.set_xticklabels(builds, fontsize=10, color=INK)
ax.set_ylabel("ns per quote (lower is better)")
ax.set_ylim(0, 640)
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
style(ax)
ax.set_title("Market feed, 30,000 quotes, all methods in one binary, quiet host (medians)",
             fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "cpu_one_binary.png"); plt.close(fig)

# ---------------------------------------------------------------- GPU (vertical, like the CPU figure)
rows = [  # label, ns/quote, color
    ("PDE, OpenCL,\nwith transfers\n(its convention)", 7.68, C_PDE),
    ("PDE, OpenCL,\nkernels +\nreadback",           4.93, C_PDE),
    ("book kernel,\nwith uploads\nand readback",                    1.12, C_BOOK),
    ("routed charts (v0.2),\nfull book,\nbucket-ordered",           0.078, C_ROUTED),
    ("book kernel,\nfull feed,\nfile order",                        0.072, C_BOOK),
    ("book kernel,\nfull feed,\nsorted by a",                       0.032, C_BOOK),
    ("recurrence kernel,\nNEAR tile",                               0.021, C_BOOK),
]
fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=200)
xs = range(len(rows))
for x, (lab, v, col) in zip(xs, rows):
    ax.bar(x, v, color=col, width=0.62, zorder=3)
    ax.text(x, v * 1.18, f"{v:g}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8, color=INK)
ax.set_yscale("log"); ax.set_ylim(0.012, 30)
ax.set_ylabel("ns per quote (log scale, lower is better)")
ax.yaxis.grid(True, color=GRID, zorder=0, which="major"); ax.set_axisbelow(True)
style(ax)
ax.set_title("GPU throughput, NVIDIA H100 PCIe, median of 300 passes, kernel-resident and with host transfers",
             fontsize=9.5, color=INK, loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_BOOK, label="book kernel (v0.3)"), Patch(color=C_ROUTED, label="routed charts (v0.2)"),
                   Patch(color=C_PDE, label="PDE table method (authors' code)")],
          frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "gpu_book_kernel.png"); plt.close(fig)
print("wrote", OUT / "cpu_one_binary.png", "and", OUT / "gpu_book_kernel.png")
