"""
fig3.py - Figure 3: immune microenvironment, checkpoint and pathway landscape.

a  immune-cell infiltration heatmap (ssGSEA, Charoentong 28 signatures)
b  selected immune populations (boxplots)
c  immune-checkpoint / cytolytic activity expression
d  hypoxia-programme scores (Buffa / Ragnum / Winter, TCGA clinical)
e  stromal & immune compartment scores
f  immune-phenotype summary (inflamed vs excluded)

Outputs: figures/Fig3.*  and results/fig3/*.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from style import (PAL, SUBTYPE_COLORS, apply_theme, box_groups, panel_label,
                   pval_text, save_fig)

FIGD = lib.FIG
RESD = os.path.join(lib.RES, "fig3")
os.makedirs(RESD, exist_ok=True)
apply_theme()

SUB = lib.merged_table()
SUBS = sorted(SUB["cluster"].unique())
NCOL = {s: SUBTYPE_COLORS.get(s, PAL["grey"]) for s in SUBS}
S2C = dict(zip(SUB["sample"], SUB["cluster"]))
MR = pd.read_csv(os.path.join(lib.PRC, "mrna.csv"), index_col=0)
samples = [s for s in SUB["sample"] if s in MR.columns]
MR = MR[samples]

CHECKPOINTS = ["PDCD1", "CD274", "PDCD1LG2", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
               "IDO1", "CD8A", "GZMA", "PRF1", "IFNG", "CXCL9", "CXCL10"]


def box_by_subtype(ax, values, ylabel, title, colors=None):
    """Box + jitter per subtype; Mann-Whitney (K=2) or Kruskal-Wallis (K>2)."""
    vals = pd.Series(values)
    data = [vals.reindex([s for s in samples if S2C.get(s) == g]).dropna().values
            for g in SUBS]
    return box_groups(ax, data, [(colors or NCOL)[g] for g in SUBS],
                      labels=SUBS, ylabel=ylabel, title=title,
                      show_n=len(SUBS) > 2)


def main():
    # ---------------- ssGSEA immune infiltration
    common = [g for g in MR.index if any(g in v for v in lib.CHAROENTONG.values())]
    sets = {k: [g for g in v if g in MR.index] for k, v in lib.CHAROENTONG.items()}
    sets = {k: v for k, v in sets.items() if len(v) >= 3}
    inf_path = os.path.join(RESD, "immune_ssgsea.csv")
    if os.path.exists(inf_path):
        inf = pd.read_csv(inf_path, index_col=0)
        inf = inf[[c for c in inf.columns if c in set(MR.columns)]]
        print(f"  immune ssGSEA loaded from cache {inf.shape}")
    else:
        inf = lib.ssgsea(MR.loc[common], sets)
        inf.to_csv(inf_path)

    # ---------------- checkpoint expression
    cps = [g for g in CHECKPOINTS if g in MR.index]
    cpm = MR.loc[cps]

    # ---------------- clinical hypoxia / purity proxies
    cl = lib.load_clinical().reindex(SUB["patient"])
    buffa = pd.to_numeric(cl.get("BUFFA_HYPOXIA_SCORE"), errors="coerce")
    ragnum = pd.to_numeric(cl.get("RAGNUM_HYPOXIA_SCORE"), errors="coerce")
    winter = pd.to_numeric(cl.get("WINTER_HYPOXIA_SCORE"), errors="coerce")
    buffa.index = cl.index
    buffa_s = pd.Series(buffa.values, index=[s for s in samples if s[:12] in set(cl.index)])

    # ---------------- stromal / immune compartment scores (ESTIMATE-like)
    strom = [g for g in ["COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1",
                         "COL6A2", "COL6A3", "FN1", "DCN", "LUM", "FAP", "PDGFRA",
                         "PDGFRB", "THY1", "ACTA2", "TAGLN", "POSTN", "MMP2",
                         "SPARC", "VIM", "FBLN1", "LOX", "TIMP1"] if g in MR.index]
    immune = [g for g in ["PTPRC", "CD2", "CD3D", "CD3E", "CD3G", "CD14", "CD19",
                          "CD79A", "CD79B", "MS4A1", "IL2RB", "LCK", "PTPRCAP",
                          "TRAC", "TRBC1", "CXCR6", "CCL5", "NKG7", "GZMA"]
              if g in MR.index]
    scores = pd.DataFrame({
        "Stromal": MR.loc[strom].mean(axis=0),
        "Immune": MR.loc[immune].mean(axis=0),
    })
    scores.to_csv(os.path.join(RESD, "compartment_scores.csv"))

    # ---------------- figure
    fig = plt.figure(figsize=(9.4, 7.6))
    gs = GridSpec(3, 3, figure=fig, hspace=0.85, wspace=0.55,
                  height_ratios=[1.15, 1, 1])

    # a heatmap
    ax = fig.add_subplot(gs[0, :])
    ordidx = sorted(range(len(samples)), key=lambda i: S2C[samples[i]])
    ordered = [samples[i] for i in ordidx]
    Z = inf[ordered]
    Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(Z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012)
    cb.set_label("z-score", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax.set_yticks(range(len(Z))); ax.set_yticklabels(Z.index, fontsize=5.6)
    ax.set_xticks([])
    pos = 0
    for s in SUBS:
        cnt = sum(1 for x in ordered if S2C[x] == s)
        ax.plot([pos, pos + cnt], [-1.6, -1.6], color=NCOL[s], lw=3,
                solid_capstyle="butt", clip_on=False)
        ax.text(pos + cnt / 2, -3.0, f"{s} (n={cnt})", ha="center", va="bottom",
                fontsize=7, color=NCOL[s], fontweight="bold", clip_on=False)
        pos += cnt
    ax.set_title("Immune-cell infiltration (ssGSEA)", fontsize=8)
    panel_label(ax, "a", y=1.01)

    # b selected immune cells
    ax = fig.add_subplot(gs[1, 0])
    sel = [s for s in ["CD8_T_cells", "Treg", "Macrophages", "NK_cells"]
           if s in inf.index]
    vals = inf.loc[sel].mean(axis=0)
    box_by_subtype(ax, vals, "ssGSEA (mean)", "Cytotoxic + regulatory cells")
    panel_label(ax, "b")

    # c checkpoint expression
    ax = fig.add_subplot(gs[1, 1])
    cpsc = cpm.loc[[g for g in ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT"]
                    if g in cpm.index]].mean(axis=0)
    box_by_subtype(ax, cpsc, "log$_2$ expression", "Immune-checkpoint score")
    panel_label(ax, "c")

    # d cytolytic activity
    ax = fig.add_subplot(gs[1, 2])
    cyto = MR.loc[[g for g in ["GZMA", "PRF1"] if g in MR.index]].mean(axis=0)
    box_by_subtype(ax, cyto, "log$_2$ expression", "Cytolytic activity (GZMA/PRF1)")
    panel_label(ax, "d")

    # e hypoxia
    ax = fig.add_subplot(gs[2, 0])
    hyp = pd.Series(index=samples, dtype=float)
    for s in samples:
        pid = s[:12]
        if pid in cl.index:
            hyp[s] = np.nanmean([buffa.get(pid, np.nan), ragnum.get(pid, np.nan),
                                 winter.get(pid, np.nan)])
    box_by_subtype(ax, hyp, "Hypoxia score", "Hypoxia programme")
    panel_label(ax, "e")

    # f stromal
    ax = fig.add_subplot(gs[2, 1])
    box_by_subtype(ax, scores["Stromal"], "mean log$_2$ expr", "Stromal compartment")
    panel_label(ax, "f")

    # g summary inflamed/excluded
    ax = fig.add_subplot(gs[2, 2])
    inf_score = scores["Immune"]
    strom_score = scores["Stromal"]
    for g in SUBS:
        idx = [s for s in samples if S2C.get(s) == g]
        ax.scatter(strom_score.reindex(idx), inf_score.reindex(idx), s=4.5,
                   color=NCOL[g], alpha=0.75, lw=0, label=g)
    ax.set_xlabel("Stromal score", fontsize=7.5)
    ax.set_ylabel("Immune score", fontsize=7.5)
    ax.set_title("Immune vs stromal compartment", fontsize=8)
    ax.legend(fontsize=6, markerscale=1.8)
    panel_label(ax, "g")

    save_fig(fig, FIGD, "Fig3")

    rows = []
    for name, v in [("Immune_infiltration_mean", inf.mean(axis=0)),
                    ("Checkpoint_score", cpsc), ("Cytolytic", cyto),
                    ("Hypoxia", hyp), ("Stromal", scores["Stromal"]),
                    ("Immune_score", scores["Immune"])]:
        vals = [pd.Series(v).reindex([s for s in samples if S2C.get(s) == g]
                                     ).dropna().values for g in SUBS]
        ok = [x for x in vals if len(x) >= 3]
        p = (float(stats.kruskal(*ok).pvalue) if len(ok) > 2
             else lib.mannwhitney(ok[0], ok[1]) if len(ok) == 2 else 1.0)
        rows.append({"feature": name,
                     **{f"{g}_mean": float(np.nanmean(x)) for g, x in zip(SUBS, vals)},
                     "p": p, "test": "kruskal" if len(ok) > 2 else "mannwhitney"})
    pd.DataFrame(rows).to_csv(os.path.join(RESD, "subtype_summary_stats.csv"),
                              index=False)
    print("DONE")


if __name__ == "__main__":
    main()
