"""
fig2.py - Figure 2: genomic and transcriptomic features of the subtypes.

a  oncoplot of the most frequently mutated genes (per subtype)
b  tumour mutational burden by subtype
c  fraction of genome altered (FGA) by subtype
d  volcano plot of differential expression (contrast pair)
e  Hallmark GSEA enrichment bubble plot (up / down in the first contrast group)
f  Hallmark pathway activity across all subtypes

Every panel generalises to an arbitrary number of subtypes; the two-group
contrast used in d/e is derived from the data by lib.contrast_pair().

Outputs: figures/Fig2.*  and results/fig2/*.csv
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
RESD = os.path.join(lib.RES, "fig2")
os.makedirs(RESD, exist_ok=True)
apply_theme()

SUB = lib.merged_table()
SUBS = sorted(SUB["cluster"].unique())
NCOL = {s: SUBTYPE_COLORS.get(s, PAL["grey"]) for s in SUBS}
S2C = dict(zip(SUB["sample"], SUB["cluster"]))
MR = pd.read_csv(os.path.join(lib.PRC, "mrna.csv"), index_col=0)
C1, C2 = lib.contrast_pair()
DEG_FILE = os.path.join(RESD, "DEG_contrast.csv")


def bh_volcano(ax, dep, c1, c2):
    ax.axhline(-np.log10(0.05), color="#BBBBBB", lw=0.6, ls="--")
    for xv in (-1, 1):
        ax.axvline(xv, color="#BBBBBB", lw=0.6, ls="--")
    dep = dep.copy()
    dep["nlp"] = -np.log10(dep["FDR"].clip(lower=1e-300))
    sig = (dep["FDR"] < 0.05) & (dep["log2FC"].abs() > 1)
    ax.scatter(dep.loc[~sig, "log2FC"], dep.loc[~sig, "nlp"], s=1.6,
               color="#D9D9D9", lw=0, rasterized=True)
    up = sig & (dep["log2FC"] > 0)
    dn = sig & (dep["log2FC"] < 0)
    ax.scatter(dep.loc[up, "log2FC"], dep.loc[up, "nlp"], s=1.8,
               color=PAL["red"], lw=0, rasterized=True)
    ax.scatter(dep.loc[dn, "log2FC"], dep.loc[dn, "nlp"], s=1.8,
               color=PAL["navy"], lw=0, rasterized=True)
    top = dep[sig].reindex(dep[sig]["nlp"].sort_values(ascending=False).index).head(7)
    for k, (_, r) in enumerate(top.iterrows()):
        ax.annotate(r["gene"], (r["log2FC"], r["nlp"]), fontsize=5.4,
                    xytext=(4 if k % 2 == 0 else -22, 2 + 5 * (k // 2)),
                    textcoords="offset points", color="#333333")
    ax.set_xlabel(f"log$_2$ fold change ({c1} vs {c2})", fontsize=7.5)
    ax.set_ylabel("$-\\log_{10}$ FDR", fontsize=7.5)
    return int(sig.sum())


def main():
    # ---------------- DEG (contrast pair)
    grp = {c: S2C.get(c) for c in MR.columns if c in S2C}
    mrna = MR.loc[:, list(grp)]
    dep = lib.diff_expr(mrna, grp, C1, C2)
    dep.to_csv(DEG_FILE, index=False)
    pd.DataFrame({"group": ["grp1", "grp2"], "subtype": [C1, C2]}).to_csv(
        os.path.join(RESD, "contrast_pair.csv"), index=False)
    print(f"  contrast {C1} vs {C2}; DEG table {dep.shape}")

    # ---------------- gene sets
    hall = lib.load_gene_sets("h.all")
    print(f"  Hallmark sets: {len(hall)}")
    gsea_path = os.path.join(RESD, "GSEA_hallmark.csv")
    if os.path.exists(gsea_path):
        gsea = pd.read_csv(gsea_path)
        print(f"  GSEA loaded from cache ({len(gsea)} rows)")
    else:
        gsea = lib.gsea_prerank(dep[["gene", "log2FC"]], hall) if hall else pd.DataFrame()
        if len(gsea):
            gsea.to_csv(gsea_path, index=False)
            print(f"  GSEA rows {len(gsea)}")

    # ---------------- mutation
    maf = pd.read_parquet(os.path.join(lib.PRC, "mut_maf.parquet"))
    maf = maf[maf["Tumor_Sample_Barcode"].isin(S2C)]
    maf["subtype"] = maf["Tumor_Sample_Barcode"].map(S2C)
    mat = (maf.assign(v=1).pivot_table(index="Hugo_Symbol",
                                       columns="Tumor_Sample_Barcode",
                                       values="v", aggfunc="max", fill_value=0))
    allsamp = list(SUB["sample"])
    mat = mat.reindex(columns=allsamp, fill_value=0)
    freq = mat.mean(axis=1).sort_values(ascending=False)
    top_g = freq.head(20).index.tolist()
    mat.to_csv(os.path.join(RESD, "mutation_matrix_full.csv"))
    pd.DataFrame({"gene": freq.index, "freq": freq.values}).to_csv(
        os.path.join(RESD, "mutation_frequency.csv"), index=False)

    # ---------------- TMB / FGA
    smp = lib.load_sample_clinical()
    smp = smp.reindex(SUB["sample"])
    fga = pd.to_numeric(smp.get("FRACTION_GENOME_ALTERED"), errors="coerce")
    # TMB = number of protein-altering somatic mutations per megabase of a
    # 30 Mb callable exome.  The MAF has already been filtered to
    # protein-altering classes, so variant-level counts ARE non-synonymous.
    tmb_manual = (maf.groupby("Tumor_Sample_Barcode").size() / 30.0)

    # =============================================== figure
    fig = plt.figure(figsize=(9.2, 7.4))
    gs = GridSpec(3, 3, figure=fig, hspace=0.85, wspace=0.5,
                  height_ratios=[1.35, 1, 1])

    # ---- a oncoplot
    ax = fig.add_subplot(gs[0, :])
    order = list(SUB["sample"])
    lab = [S2C[s] for s in order]
    sub_order = sorted(range(len(order)), key=lambda i: (lab[i], i))
    order = [order[i] for i in sub_order]
    mm = mat.reindex(columns=order).loc[top_g]
    cls = {"Missense_Mutation": PAL["navy"], "Nonsense_Mutation": PAL["red"],
           "Frame_Shift_Del": PAL["salmon"], "Frame_Shift_Ins": PAL["orange"],
           "In_Frame_Del": PAL["teal"], "Splice_Site": PAL["purple"],
           "Translation_Start_Site": PAL["purple"], "Nonstop_Mutation": PAL["pink"]}
    vc = maf.pivot_table(index="Hugo_Symbol", columns="Tumor_Sample_Barcode",
                         values="Variant_Classification", aggfunc="first")
    vc = vc.reindex(columns=order).reindex(top_g)
    for gi, g in enumerate(top_g):
        for si, s in enumerate(order):
            if mm.loc[g, s] == 1:
                v = vc.loc[g, s]
                ax.add_patch(plt.Rectangle((si, gi - 0.42), 0.86, 0.84,
                             color=cls.get(v, PAL["slate"]), lw=0))
    ax.set_xlim(-1, len(order) + 16)
    ax.set_ylim(len(top_g) - 0.4, -7.0)
    ax.set_yticks(range(len(top_g)))
    ax.set_yticklabels([f"{g}" for g in top_g], fontsize=6)
    for gi, g in enumerate(top_g):
        ax.text(len(order) + 3, gi, f"{freq[g]*100:.0f}%", fontsize=5.6,
                va="center", ha="left", color="#555555")
    ax.set_xticks([])
    pos = 0
    for s in SUBS:
        cnt = sum(1 for x in order if S2C[x] == s)
        ax.plot([pos, pos + cnt], [-1.8, -1.8], color=NCOL[s], lw=3,
                solid_capstyle="butt", clip_on=False)
        ax.text(pos + cnt / 2, -4.6, f"{s} (n={cnt})", ha="center", va="bottom",
                fontsize=7, color=NCOL[s], fontweight="bold", clip_on=False)
        pos += cnt
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=3.2, color=c, label=l)
               for l, c in cls.items() if l in set(maf["Variant_Classification"])]
    ax.legend(handles=handles, fontsize=5.4, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.30), frameon=False)
    ax.set_title("Somatic mutation landscape", fontsize=8, pad=6)
    panel_label(ax, "a", y=1.02)

    # ---- b TMB  (any number of subtypes)
    ax = fig.add_subplot(gs[1, 0])
    data = [tmb_manual[[s for s in SUB["sample"] if S2C[s] == g]].dropna().values
            for g in SUBS]
    box_groups(ax, data, [NCOL[g] for g in SUBS], labels=SUBS,
               ylabel="Mutation burden\n(per Mb, log scale)",
               title="Tumour mutational burden")
    # one hypermutated tumour (126 mutations per Mb) otherwise compresses both
    # boxes onto the baseline; log scale keeps the whole distribution visible
    ax.set_yscale("log")
    ax.set_ylim(0.03, 200)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"], fontsize=6.5)
    panel_label(ax, "b")

    # ---- c FGA
    ax = fig.add_subplot(gs[1, 1])
    data = [fga[[s for s in SUB["sample"] if S2C[s] == g]].dropna().values
            for g in SUBS]
    box_groups(ax, data, [NCOL[g] for g in SUBS], labels=SUBS,
               ylabel="Fraction genome altered", title="Copy-number burden")
    panel_label(ax, "c")

    # ---- d volcano
    ax = fig.add_subplot(gs[1, 2])
    n_sig = bh_volcano(ax, dep, C1, C2)
    ax.set_title("Differential expression", fontsize=8)
    ax.text(0.03, 0.96, f"{n_sig} genes\nFDR<0.05 & |log2FC|>1",
            transform=ax.transAxes, fontsize=6, va="top")
    panel_label(ax, "d")

    # ---- e GSEA bubble
    ax = fig.add_subplot(gs[2, :2])
    if len(gsea):
        g2 = gsea.copy()
        g2["NES"] = pd.to_numeric(g2["NES"], errors="coerce")
        g2["FDR q-val"] = pd.to_numeric(g2.get("FDR q-val"), errors="coerce")
        g2 = g2.dropna(subset=["NES"]).reindex(
            g2["NES"].abs().sort_values(ascending=False).index).head(16)
        g2 = g2.sort_values("NES")
        col = [PAL["red"] if v > 0 else PAL["navy"] for v in g2["NES"]]
        sz = (-np.log10(g2["FDR q-val"].clip(lower=1e-6)) * 8).clip(4, 70)
        ax.scatter(g2["NES"], np.arange(len(g2)), s=sz, color=col, alpha=0.85, lw=0)
        ax.set_yticks(np.arange(len(g2)))
        ax.set_yticklabels([t.replace("HALLMARK_", "").replace("_", " ").title()
                            for t in g2["Term"]], fontsize=5.8)
        ax.axvline(0, color="#BBBBBB", lw=0.6)
        ax.set_xlabel(f"Normalised enrichment score ({C1} vs {C2})", fontsize=7.5)
        ax.set_title("Hallmark pathway activity (GSEA)", fontsize=8)
        ax.legend(handles=[plt.Line2D([], [], marker="o", ls="none", ms=4,
                                      color=PAL["red"], label=f"Up in {C1}"),
                           plt.Line2D([], [], marker="o", ls="none", ms=4,
                                      color=PAL["navy"], label=f"Up in {C2}")],
                  fontsize=6, loc="lower right")
    else:
        ax.axis("off")
    panel_label(ax, "e")

    # ---- f Hallmark activity across all subtypes (heatmap, K-agnostic)
    ax = fig.add_subplot(gs[2, 2])
    sets = lib.hallmark_pick(hall, {
        "E2F targets": ("e2f",),
        "G2M checkpoint": ("g2m",),
        "EMT": ("epithelial", "mesenchymal"),
        "Hypoxia": ("hypoxia",),
        "Inflammatory response": ("inflammatory",),
        "Estrogen response (early)": ("estrogen", "early")})
    if sets:
        sub_mat = mrna.loc[[g for g in mrna.index
                            if any(g in s for s in sets.values())]]
        sc = lib.ssgsea(sub_mat, sets)
        sc.to_csv(os.path.join(RESD, "hallmark_scores.csv"))
        rows = []
        for term in sc.index:
            vals = [sc.loc[term, [s for s in sc.columns if S2C.get(s) == g]].values
                    for g in SUBS]
            vals = [v[~pd.isna(v)] for v in vals]
            ok = [v for v in vals if len(v) >= 3]
            p = float(stats.kruskal(*ok).pvalue) if len(ok) > 1 else 1.0
            rows.append({"term": term, **{g: np.nanmean(v) for g, v in zip(SUBS, vals)},
                         "p": p})
        r = pd.DataFrame(rows).set_index("term")
        r.to_csv(os.path.join(RESD, "hallmark_by_subtype.csv"))
        Z = r[SUBS].sub(r[SUBS].mean(axis=1), axis=0).div(
            r[SUBS].std(axis=1).replace(0, np.nan), axis=0).fillna(0)
        im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(range(len(SUBS)))
        ax.set_xticklabels(SUBS, fontsize=6.5)
        ax.set_yticks(range(len(Z)))
        ax.set_yticklabels([t.title() for t in Z.index], fontsize=5.8)
        for i, p in enumerate(r["p"]):
            ax.text(len(SUBS) - 0.35, i, pval_text(p), fontsize=5.6, va="center",
                    ha="left", color="#1B1B1B")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.16)
        cb.ax.tick_params(labelsize=5.6)
        cb.set_label("ssGSEA z", fontsize=6)
        ax.set_title("Pathway activity by subtype", fontsize=8)
    panel_label(ax, "f", x=-0.30)

    save_fig(fig, FIGD, "Fig2")
    print("DONE")


if __name__ == "__main__":
    main()
