"""
style.py - shared Cell-press-style plotting theme for the BRCA project.

Design rules (applied to every figure in the manuscript):
  * white background, no chart junk, thin axis lines
  * Arial (falls back to Helvetica/DejaVu Sans), explicit point sizes
  * lowercase panel labels a, b, c ... bold, 12 pt
  * palette: NPG-inspired low-saturation colours + grey scale
  * every axis label / tick / legend entry has an explicit fontsize
  * output = editable vector (PDF with TrueType text + SVG with real text)
    plus a 300-dpi PNG preview
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.use("Agg")

# ---------------------------------------------------------------- palette
PAL = {
    "red": "#E64B35", "blue": "#4DBBD5", "teal": "#00A087", "navy": "#3C5488",
    "salmon": "#F39B7F", "slate": "#8491B4", "green": "#91D1C2",
    "brown": "#7E6148", "grey": "#B09C85", "dark": "#1B1B1B",
    "orange": "#EFC000", "purple": "#7E57C2", "pink": "#E377C2",
}
SUB_COLORS = ["#E64B35", "#3C5488", "#00A087", "#F39B7F", "#7E57C2", "#EFC000"]
# subtype palette (BC1..BCn)
SUBTYPE_COLORS = {"BC1": "#E64B35", "BC2": "#3C5488", "BC3": "#00A087",
                  "BC4": "#F39B7F", "BC5": "#7E57C2"}
GRP2 = ["#E64B35", "#3C5488"]

FONTS = ["Arial", "Helvetica", "DejaVu Sans"]


def apply_theme(base=7.5):
    """base = default font size in points (Cell figures ~6-8 pt at final size)."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#1B1B1B",
        "axes.labelcolor": "#1B1B1B",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.color": "#1B1B1B",
        "ytick.color": "#1B1B1B",
        "legend.fontsize": base - 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # TrueType -> editable text in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # real <text> elements in SVG
        # mathtext ($...$) defaults to DejaVu Sans, which would mix a second
        # typeface into otherwise Arial figures -> force Arial for math too
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.tt": "Arial",
        "mathtext.cal": "Arial:italic",
        "mathtext.default": "regular",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "#1B1B1B",
        "axes.grid": False,
        "errorbar.capsize": 0,
        "boxplot.showfliers": False,
    })


def panel_label(ax, letter, x=-0.13, y=1.06, size=None):
    """Lowercase bold panel label placed relative to axes coordinates."""
    if size is None:
        size = mpl.rcParams["font.size"] + 3.5
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def pval_text(p):
    if p != p:
        return "ns"
    return ("***" if p < 1e-3 else "**" if p < 1e-2 else
            "*" if p < 5e-2 else "ns")


def save_fig(fig, outdir, name, formats=("pdf", "svg", "png"), dpi=300):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for ext in formats:
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p, format=ext, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------- primitives
def despine(ax, left=False, bottom=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
    return ax


def jitter(n, width=0.16, seed=0):
    import numpy as np
    return np.random.default_rng(seed).uniform(-width, width, n)


def scalebar(ax, size, color="#1B1B1B", lw=1.0, label=None, loc=(0.02, 0.03),
             label_size=None):
    """Compact 'n = ' style annotation box for a panel."""
    if label:
        ax.text(loc[0], loc[1], label, transform=ax.transAxes,
                fontsize=label_size or mpl.rcParams["font.size"] - 0.5,
                color=color, va="bottom", ha="left")


def legend_strip(ax, items, colors, y=1.12, x=0.0, marker="s", ncol=6,
                 size=None, gap=0.085):
    """Manual legend drawn as coloured squares + text (Cell style legends)."""
    from matplotlib.lines import Line2D
    sz = size or mpl.rcParams["font.size"] - 0.5
    handles = [Line2D([], [], marker=marker, linestyle="none", markersize=3.4,
                      markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in zip(items, colors)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y),
              ncol=ncol, frameon=False, fontsize=sz, handletextpad=0.3,
              columnspacing=0.9, borderaxespad=0)


def add_n(ax, groups, values, y, dy=None):
    """Print n = .. above each group (Cell style)."""
    import numpy as np
    for i, g in enumerate(groups):
        ax.text(i, y, f"n={len(values[i])}", ha="center", va="bottom",
                fontsize=mpl.rcParams["font.size"] - 1.0, color="#4D4D4D")


def box_groups(ax, values, colors, labels=None, ylabel=None, ymax=None,
               test="auto", seed=0, jitter_w=0.13, point_size=1.2,
               title=None, show_n=True):
    """Box + jittered points for an arbitrary number of groups.

    Statistical test: Mann-Whitney U for two groups, Kruskal-Wallis for
    more than two; the resulting p-value is printed above the boxes.
    Works for K = 2..5 so that no panel silently assumes a two-subtype split.
    """
    import numpy as np
    from scipy import stats
    k = len(values)
    vals = [np.asarray(v, float) for v in values]
    vals = [v[~np.isnan(v)] for v in vals]
    bp = ax.boxplot(vals, widths=0.52, patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(c)
        b.set_alpha(0.55)
        b.set_edgecolor(c)
        b.set_linewidth(0.8)
    for m in bp["medians"]:
        m.set_color("#1B1B1B")
        m.set_linewidth(0.9)
    for i, v in enumerate(vals):
        ax.scatter(i + 1 + jitter(len(v), jitter_w, seed + i), v, s=point_size,
                   color="#4D4D4D", alpha=0.40, lw=0, rasterized=True)
    ytop = ymax if ymax is not None else (
        max([np.nanmax(v) for v in vals if len(v)] or [1]) * 1.28)
    ax.set_ylim(0 if min([np.nanmin(v) for v in vals if len(v)] or [0]) >= 0
                else None, ytop)
    if test != "none" and all(len(v) >= 3 for v in vals):
        if k == 2:
            p = float(stats.mannwhitneyu(vals[0], vals[1],
                                         alternative="two-sided").pvalue)
        else:
            p = float(stats.kruskal(*vals).pvalue)
        ax.plot([1, k], [ytop * 0.93] * 2, color="#1B1B1B", lw=0.7)
        ax.text((1 + k) / 2, ytop * 0.945, pval_text(p), ha="center", fontsize=7)
    if show_n:
        for i, v in enumerate(vals):
            ax.text(i + 1, ytop * 0.02, f"n={len(v)}", ha="center", va="bottom",
                    fontsize=5.6, color="#666666")
    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels(labels if labels is not None else
                       [f"G{i+1}" for i in range(k)], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7.5)
    if title:
        ax.set_title(title, fontsize=8)
    for i, v in enumerate(vals):
        if len(v) == 0:
            ax.text(i + 1, 0.5, "no data", ha="center", fontsize=6,
                    transform=ax.get_xaxis_transform())
    return bp

