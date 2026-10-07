"""
figS.py - Supplementary Figures S1-S12.

Numbering follows the order of first citation in the manuscript (NC style):
S1  data overview: sample counts per omics layer and data distributions
S2  consensus matrix and stability diagnostics (K selection)
S3  clinical characteristics of the subtypes (age, stage, grade, histology)
S4  subtype survival across all four annotated endpoints (OS/PFS/DSS/DFS)
S5  extended somatic mutation landscape (genes mutated in >= 3%)
S6  differential-expression summary and Hallmark GSEA bubble plot
S7  full immune infiltration profile (25 Charoentong signatures)
S8  distribution of the 101 model C-indices with cross-validation spread
S9  risk-score survival for OS and DFS, plus HR forest per endpoint
S10 risk score across clinical subgroups
S11 CPTAC protein / phosphosite effect-size distributions
S12 single-cell quality metrics and cell-type composition

Outputs: figures/FigS1..FigS12.*  and results/supp/*.csv
NOTE: functions appear in file in historical order; the def name equals the
output figure number (def S11 -> FigS11), so the dispatch loop still works.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from style import PAL, SUBTYPE_COLORS, apply_theme, panel_label, pval_text, save_fig

FIGD = lib.FIG
RESD = os.path.join(lib.RES, "supp")
os.makedirs(RESD, exist_ok=True)
apply_theme()

SUB = lib.merged_table()
SUBS = sorted(SUB["cluster"].unique())
NCOL = {s: SUBTYPE_COLORS.get(s, PAL["grey"]) for s in SUBS}
S2C = dict(zip(SUB["sample"], SUB["cluster"]))
C1, C2 = lib.contrast_pair()


def box(ax, data, labels, colors, ylabel, title, jitter=0.14, ptest=True):
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(c); b.set_alpha(0.5); b.set_edgecolor(c); b.set_lw(0.8)
    for m in bp["medians"]:
        m.set_color("#1B1B1B"); m.set_lw(0.9)
    for i, d in enumerate(data):
        d = np.asarray(d, float); d = d[~np.isnan(d)]
        ax.scatter(i + 1 + np.random.default_rng(i).uniform(-jitter, jitter, len(d)),
                   d, s=1.1, color="#4D4D4D", alpha=0.35, lw=0, rasterized=True)
    if ptest and len(data) == 2 and all(len(np.asarray(d)[~np.isnan(np.asarray(d, float))]) > 3 for d in data):
        dd = [np.asarray(d, float) for d in data]
        ymax = np.nanmax([np.nanmax(d) for d in dd])
        ymin = np.nanmin([np.nanmin(d) for d in dd])
        rng = ymax - ymin if ymax > ymin else 1
        p = lib.mannwhitney(*dd)
        ax.plot([1, 2], [ymax + rng * 0.1] * 2, color="#1B1B1B", lw=0.7)
        ax.text(1.5, ymax + rng * 0.11, pval_text(p), ha="center", fontsize=7)
        ax.set_ylim(ymin - rng * 0.08, ymax + rng * 0.25)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title(title, fontsize=8)


def S1():
    fig = plt.figure(figsize=(10.4, 5.6))
    gs = GridSpec(2, 5, figure=fig, hspace=0.6, wspace=0.55, height_ratios=[1, 1.15])
    layers = {}
    for name, fn in [("mRNA", "mrna.csv"), ("lncRNA", "lncrna.csv"),
                     ("miRNA", "mirna.csv"), ("Methylation", "methyl_top10000.csv"),
                     ("Mutation", "mut_matrix.csv")]:
        p = os.path.join(lib.PRC, fn)
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0, nrows=5)
            layers[name] = len([c for c in d.columns])
    ax = fig.add_subplot(gs[0, :3])
    ks = list(layers)
    colsets = {}
    for name, fn in [("mRNA", "mrna.csv"), ("lncRNA", "lncrna.csv"),
                     ("miRNA", "mirna.csv"), ("Methylation", "methyl_top10000.csv"),
                     ("Mutation", "mut_matrix.csv")]:
        p = os.path.join(lib.PRC, fn)
        if os.path.exists(p):
            with open(p) as f:
                colsets[name] = set(f.readline().rstrip("\n").split(",")[1:])
    n_feat = {}
    for name, fn in [("mRNA", "mrna.csv"), ("lncRNA", "lncrna.csv"),
                     ("miRNA", "mirna.csv"), ("Methylation", "methyl_top10000.csv"),
                     ("Mutation", "mut_matrix.csv")]:
        p = os.path.join(lib.PRC, fn)
        if os.path.exists(p):
            with open(p) as f:
                n_feat[name] = sum(1 for _ in f) - 1
    ax.bar(np.arange(len(ks)), [layers[k] for k in ks], color=PAL["navy"], width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, layers[k] + 8, f"{layers[k]}", ha="center", fontsize=6.5)
    n_common = len(set.intersection(*colsets.values()))
    print(f"  S1: intersection of {len(colsets)} layers = {n_common} samples")
    ax.axhline(n_common, ls="--", lw=0.9, color=PAL["red"], zorder=1)
    ax.text(0.02, 0.99,
            f"{n_common} tumours with all five layers (clustering cohort)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, color=PAL["red"])
    ax.set_ylim(0, max(layers.values()) * 1.18)
    ax.set_xticks(np.arange(len(ks))); ax.set_xticklabels(ks, rotation=25,
                                                          ha="right", fontsize=7)
    ax.set_ylabel("Samples", fontsize=7.5)
    ax.set_title("Samples per omics layer", fontsize=8)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 3:])
    # features actually entering consensus clustering (after MAD / frequency
    # selection in 10_cluster.py), not the raw per-layer inputs
    import pickle as _pkl
    _pklp = os.path.join(lib.RES, "cluster", "feature_layers.pkl")
    if os.path.exists(_pklp):
        with open(_pklp, "rb") as _f:
            _fl = _pkl.load(_f)
        sel_feat = {k: int(v.shape[1]) for k, v in _fl["layers"].items()}
    else:
        sel_feat = {k: min(n_feat.get(k, 0), 1) for k in ks}
    x = np.arange(len(ks))
    ax.bar(x, [sel_feat.get(k, 0) for k in ks], color=PAL["teal"], width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, sel_feat.get(k, 0) * 1.05, f"{sel_feat.get(k,0):,}", ha="center",
                fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(10, 1e5)
    ax.set_xticks(x); ax.set_xticklabels(ks, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Features (log scale)", fontsize=7.5)
    ax.set_title("Selected features per layer used for clustering", fontsize=8)
    panel_label(ax, "b")

    mr = pd.read_csv(os.path.join(lib.PRC, "mrna.csv"), index_col=0)
    for j, (nm, fn) in enumerate([("mRNA", "mrna.csv"), ("lncRNA", "lncrna.csv"),
                                  ("miRNA", "mirna.csv")]):
        ax = fig.add_subplot(gs[1, j])
        d = pd.read_csv(os.path.join(lib.PRC, fn), index_col=0)
        v = d.values.ravel()
        v = v[np.isfinite(v)]
        ax.hist(v, bins=80, color=PAL["slate"], lw=0)
        ax.set_xlabel("log$_2$ expression", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.set_title(nm, fontsize=8)
        panel_label(ax, "cde"[j])
    ax = fig.add_subplot(gs[1, 3])
    d = pd.read_csv(os.path.join(lib.PRC, "methyl_top10000.csv"), index_col=0)
    v = d.values.ravel(); v = v[np.isfinite(v)]
    ax.hist(v, bins=60, color=PAL["salmon"], lw=0)
    ax.set_xlabel("Beta value", fontsize=7); ax.set_ylabel("Count", fontsize=7)
    ax.set_title("Methylation", fontsize=8)
    panel_label(ax, "f")

    # the mutation layer is binary (0/1), so a raw-value histogram is
    # uninformative; show the per-sample mutation burden within the 666-sample
    # clustering cohort instead
    ax = fig.add_subplot(gs[1, 4])
    mut = pd.read_csv(os.path.join(lib.PRC, "mut_matrix.csv"), index_col=0)
    if os.path.exists(_pklp):
        coh = [s for s in _fl["samples"] if s in mut.columns]
        burden = mut[coh].sum(axis=0)
        ax.hist(burden, bins=30, color=PAL["red"], lw=0)
        ax.axvline(float(burden.median()), ls="--", lw=0.8, color="#444444")
        ax.text(0.97, 0.96, f"median {burden.median():.0f} genes",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                color="#444444")
    ax.set_xlabel("Mutated genes per tumour", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.set_title("Mutation burden", fontsize=8)
    panel_label(ax, "g")
    save_fig(fig, FIGD, "FigS1")


def S2():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    C = None
    with open(os.path.join(lib.RES, "cluster", "feature_layers.pkl"), "rb") as f:
        fl = pickle.load(f)
    sub = pd.read_csv(os.path.join(lib.RES, "cluster", "subtypes.csv"))
    order = np.argsort(sub["cluster"].values)
    Cm = pd.read_csv(os.path.join(lib.RES, "cluster",
                                  f"consensus_K{len(SUBS)}.csv"), header=None).values
    ax = axes[1]
    im = ax.imshow(Cm[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Consensus matrix", fontsize=8)
    panel_label(ax, "b", x=-0.07)

    ks = pd.read_csv(os.path.join(lib.RES, "cluster", "k_selection.csv"))
    ax = axes[0]
    ax.plot(ks["K"], ks["silhouette"], "-o", color=PAL["red"], ms=3, label="Silhouette")
    ax.plot(ks["K"], ks["mean_within_algo_agreement"], "-s", color=PAL["navy"],
            ms=3, label="Algorithm agreement")
    ax.set_xlabel("K", fontsize=7.5); ax.set_ylabel("Score", fontsize=7.5)
    ax.set_xticks(ks["K"])
    ax.legend(fontsize=6.5)
    ax.set_title("Stability diagnostics", fontsize=8)
    panel_label(ax, "a")
    save_fig(fig, FIGD, "FigS2")


def S3():
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.0))
    d = SUB.copy()
    d["age_n"] = pd.to_numeric(d["age"], errors="coerce")
    d["stage_n"] = d["stage"].str.extract(r"STAGE ([IV]+)")[0].map(
        {"I": 1, "II": 2, "III": 3, "IV": 4})
    # NOTE: the TCGA pan-can atlas 2018 cBioPortal study carries no tumour
    # GRADE attribute, so the histology composition (TUMOR_TYPE) is shown
    # in panel c instead of a grade boxplot.
    for k, (ax, (col, lab, title)) in enumerate(zip(axes[:2], [
            ("age_n", "Age (years)", "Age"), ("stage_n", "Stage", "AJCC stage")])):
        data = [d.loc[d["cluster"] == g, col].dropna().values for g in SUBS]
        box(ax, data, SUBS, [NCOL[g] for g in SUBS], lab, title)
        panel_label(ax, "ab"[k])
    ax = axes[2]
    mapping = {"Infiltrating Ductal Carcinoma": "Ductal",
               "Infiltrating Lobular Carcinoma": "Lobular"}
    hist = (d.assign(h=d["TUMOR_TYPE"].map(mapping).fillna("Other"))
              .groupby(["cluster", "h"]).size().unstack(fill_value=0)
              .reindex(SUBS))
    order = [c for c in ["Ductal", "Lobular", "Other"] if c in hist.columns]
    hist = hist[order]
    frac = hist.div(hist.sum(axis=1), axis=0)
    hcol = {"Ductal": "#3C5488", "Lobular": "#E64B35", "Other": "#B09C85"}
    bottom = np.zeros(len(SUBS))
    for cat in order:
        ax.bar(np.arange(len(SUBS)), frac[cat].values, bottom=bottom,
               color=hcol[cat], width=0.55, edgecolor="white", lw=0.5,
               label=cat)
        for i, (v, n) in enumerate(zip(frac[cat].values, hist[cat].values)):
            if v > 0.07:
                ax.text(i, bottom[i] + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=6,
                        color="white" if cat != "Other" else "#1B1B1B")
        bottom += frac[cat].values
    table = hist.values
    from scipy.stats import chi2_contingency
    p = chi2_contingency(table)[1]
    ax.text(0.03, 0.97, f"\u03c7\u00b2 p = {p:.1e}",
            transform=ax.transAxes, fontsize=6.5, va="top", color="#1B1B1B")
    ax.set_xticks(np.arange(len(SUBS))); ax.set_xticklabels(SUBS, fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), ncol=3, handlelength=1.0)
    ax.set_title("Histological type", fontsize=8)
    panel_label(ax, "c")
    ax = axes[3]
    histp = d["pam50"].value_counts()
    ax.barh(np.arange(len(histp))[::-1], histp.values, color=PAL["slate"], height=0.6)
    ax.set_yticks(np.arange(len(histp))[::-1]); ax.set_yticklabels(histp.index, fontsize=6.5)
    ax.set_xlabel("Samples", fontsize=7.5); ax.set_title("PAM50 classes", fontsize=8)
    for i, v in enumerate(histp.values):
        ax.text(v + 3, len(histp) - 1 - i, str(v), va="center", fontsize=6)
    panel_label(axes[3], "d")
    save_fig(fig, FIGD, "FigS3")


def S5():
    maf = pd.read_parquet(os.path.join(lib.PRC, "mut_maf.parquet"))
    maf = maf[maf["Tumor_Sample_Barcode"].isin(S2C)]
    maf["subtype"] = maf["Tumor_Sample_Barcode"].map(S2C)
    mat = maf.assign(v=1).pivot_table(index="Hugo_Symbol",
                                      columns="Tumor_Sample_Barcode", values="v",
                                      aggfunc="max", fill_value=0)
    mat = mat.reindex(columns=list(SUB["sample"]), fill_value=0)
    freq = mat.mean(axis=1).sort_values(ascending=False)
    sel = freq[freq >= 0.03].index
    fig, ax = plt.subplots(figsize=(9.2, 0.16 * len(sel) + 1.6))
    order = np.argsort([S2C[s] for s in mat.columns])
    mm = mat.iloc[:, order].loc[sel]
    ax.imshow(mm.values, aspect="auto", cmap="Greys", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=5)
    ax.set_xticks([])
    pos = 0
    for s in SUBS:
        cnt = sum(1 for i in order if S2C[mat.columns[i]] == s)
        ax.plot([pos, pos + cnt], [-2.0, -2.0], color=NCOL[s], lw=4,
                solid_capstyle="butt", clip_on=False)
        ax.text(pos + cnt / 2, -5.0, f"{s} (n={cnt})", ha="center", fontsize=7,
                color=NCOL[s], fontweight="bold", clip_on=False)
        pos += cnt
    ax.set_xlim(-1, len(order))
    ax.set_title(f"Somatic mutations in genes mutated in >=3% of samples "
                 f"(n={len(sel)} genes)", fontsize=8)
    panel_label(ax, "a", x=-0.055, y=1.01)
    pd.DataFrame({"gene": freq.index, "freq": freq.values}).to_csv(
        os.path.join(RESD, "mutation_frequency_all.csv"), index=False)
    save_fig(fig, FIGD, "FigS5")


def S6():
    dep = pd.read_csv(os.path.join(lib.RES, "fig2", "DEG_contrast.csv"))
    dep["nlp"] = -np.log10(dep["FDR"].clip(lower=1e-300))
    fig = plt.figure(figsize=(9.2, 3.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    sig = (dep["FDR"] < 0.05) & (dep["log2FC"].abs() > 1)
    up = int((sig & (dep["log2FC"] > 0)).sum())
    dn = int((sig & (dep["log2FC"] < 0)).sum())
    ax.bar([0, 1], [up, dn], color=[PAL["red"], PAL["navy"]], width=0.55)
    for i, v in enumerate([up, dn]):
        ax.text(i, v * 1.02, str(v), ha="center", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Up in {C1}", f"Up in {C2}"], fontsize=7)
    ax.set_ylabel("Genes", fontsize=7.5)
    ax.set_title("Differentially expressed genes", fontsize=8)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1:])
    g = pd.read_csv(os.path.join(lib.RES, "fig2", "GSEA_hallmark.csv"))
    g = g.dropna(subset=["NES"]).copy()
    g["NES"] = pd.to_numeric(g["NES"], errors="coerce")
    g["FDR q-val"] = pd.to_numeric(g["FDR q-val"], errors="coerce")
    g = g.reindex(g["NES"].abs().sort_values(ascending=False).index).head(25)
    g = g.sort_values("NES")
    cols = [PAL["red"] if v > 0 else PAL["navy"] for v in g["NES"]]
    ax.scatter(g["NES"], np.arange(len(g)), s=(-np.log10(g["FDR q-val"].clip(lower=1e-8)) * 9).clip(4, 80),
               color=cols, lw=0)
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels(g["Term"].str.replace("HALLMARK_", "").str.replace("_", " "),
                       fontsize=5.4)
    ax.axvline(0, color="#BBBBBB", lw=0.6)
    ax.set_xlabel(f"NES ({C1} vs {C2})", fontsize=7.5)
    ax.set_title("Hallmark GSEA (top 25)", fontsize=8)
    panel_label(ax, "b")
    save_fig(fig, FIGD, "FigS6")


def S7():
    inf = pd.read_csv(os.path.join(lib.RES, "fig3", "immune_ssgsea.csv"), index_col=0)
    inf = inf[[c for c in inf.columns if c in S2C]]
    rows = []
    for t in inf.index:
        a = inf.loc[t, [c for c in inf.columns if S2C[c] == C1]]
        b = inf.loc[t, [c for c in inf.columns if S2C[c] == C2]]
        rows.append({"term": t, C1: a.mean(), C2: b.mean(),
                     "p": lib.mannwhitney(a, b)})
    r = pd.DataFrame(rows).sort_values(C1)
    fig, ax = plt.subplots(figsize=(8.6, 0.22 * len(r) + 1.4))
    y = np.arange(len(r))[::-1]
    ax.barh(y - 0.19, r[C1], height=0.36, color=SUBTYPE_COLORS[C1])
    ax.barh(y + 0.19, r[C2], height=0.36, color=SUBTYPE_COLORS[C2])
    ax.set_yticks(y); ax.set_yticklabels(r["term"], fontsize=5.6)
    for i, pv in enumerate(r["p"]):
        ax.text(max(r[C1].iloc[i], r[C2].iloc[i]) + 0.002,
                len(r) - 1 - i, pval_text(pv), fontsize=4.8, va="center")
    ax.set_xlabel("ssGSEA score", fontsize=7.5)
    ax.set_title("Immune infiltration across 25 signatures", fontsize=8)
    ax.legend([C1, C2], fontsize=6.5)
    panel_label(ax, "a", x=-0.20, y=1.01)
    r.to_csv(os.path.join(RESD, "immune_by_subtype.csv"), index=False)
    save_fig(fig, FIGD, "FigS7")


def S8():
    s = pd.read_csv(os.path.join(lib.RES, "model", "model_scores_101.csv"))
    s = s.dropna(subset=["mean_C"]).sort_values("mean_C", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    ax = axes[0]
    ax.errorbar(np.arange(len(s)), s["mean_C"], yerr=s["sd_C"], fmt="none",
                ecolor="#CCCCCC", lw=0.5)
    ax.plot(np.arange(len(s)), s["mean_C"], ".", color=PAL["navy"], ms=2.2)
    ax.axhline(0.5, color="#999999", lw=0.6, ls="--")
    ax.set_xlabel("Model rank", fontsize=7.5); ax.set_ylabel("C-index", fontsize=7.5)
    ax.set_title("All 101 models ranked", fontsize=8)
    panel_label(ax, "a")
    ax = axes[1]
    ax.hist(s["mean_C"], bins=28, color=PAL["slate"], lw=0)
    ax.axvline(s["mean_C"].max(), color=PAL["red"], lw=1.0,
               label=f"best = {s['mean_C'].max():.3f}")
    ax.axvline(0.5, color="#999999", lw=0.7, ls="--", label="random")
    ax.set_xlabel("Mean CV C-index", fontsize=7.5); ax.set_ylabel("Models", fontsize=7.5)
    ax.set_title("C-index distribution", fontsize=8)
    ax.legend(fontsize=6.5)
    panel_label(ax, "b")
    save_fig(fig, FIGD, "FigS8")


def S9():
    rt = pd.read_csv(os.path.join(lib.RES, "model", "risk_table.csv"))
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 3.0))
    rt["age_n"] = pd.to_numeric(rt["age"], errors="coerce")
    rt["stage_n"] = rt["stage"].str.extract(r"STAGE ([IV]+)")[0].map(
        {"I": 1, "II": 2, "III": 3, "IV": 4})
    box(axes[0], [rt.loc[rt["risk_group"] == g, "risk"].values
                  for g in ["Low risk", "High risk"]], ["Low", "High"],
        [PAL["navy"], PAL["red"]], "Risk score", "Risk groups")
    box(axes[1], [rt.loc[rt["cluster"] == g, "risk"].values for g in SUBS], SUBS,
        [NCOL[g] for g in SUBS], "Risk score", "By subtype")
    sub = rt.dropna(subset=["stage_n"])
    box(axes[2], [sub.loc[sub["stage_n"] == s, "risk"].values
                  for s in sorted(sub["stage_n"].unique())],
        [f"Stage {int(s)}" for s in sorted(sub["stage_n"].unique())],
        [PAL["slate"]] * 4, "Risk score", "By stage")
    sub = rt.dropna(subset=["age_n"])
    sub["age_bin"] = pd.cut(sub["age_n"], [0, 40, 50, 60, 70, 120],
                            labels=["<40", "40-49", "50-59", "60-69", "70+"])
    box(axes[3], [sub.loc[sub["age_bin"] == b, "risk"].dropna().values
                  for b in sub["age_bin"].cat.categories],
        list(sub["age_bin"].cat.categories), [PAL["slate"]] * 5,
        "Risk score", "By age")
    for k, ax in enumerate(axes):
        panel_label(ax, "abcd"[k])
    save_fig(fig, FIGD, "FigS9")


def S11():
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for k, (ax, (nm, fn, col)) in enumerate(zip(axes, [
            ("Proteins", "differential_proteins.csv", PAL["red"]),
            ("Phosphosites", "differential_phosphosites.csv", PAL["navy"]),
            ("Phospho gene", "differential_phospho_genes.csv", PAL["teal"])])):
        p = os.path.join(lib.RES, "fig6", fn)
        if os.path.exists(p):
            d = pd.read_csv(p)
            ax.hist(d["log2FC"].dropna(), bins=60, color=col, lw=0)
            ax.axvline(0, color="#999999", lw=0.6, ls="--")
            ax.set_xlabel(f"log$_2$ FC ({C1}/{C2})", fontsize=7.5)
            ax.set_ylabel("Count", fontsize=7.5)
            ax.set_title(nm, fontsize=8)
        panel_label(ax, "abc"[k])
    save_fig(fig, FIGD, "FigS11")


def S12():
    um = pd.read_csv(os.path.join(lib.RES, "sc", "umap_all.csv"))
    fig = plt.figure(figsize=(9.2, 3.2))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(um["UMAP1"], um["UMAP2"], s=0.08, c=um["donor"].astype("category").cat.codes,
               cmap="tab20", lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title("By donor (26 samples)", fontsize=8)
    panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(um["UMAP1"], um["UMAP2"], s=0.08, c=um["call"].map(
        {"cancer": PAL["red"], "normal": PAL["navy"],
         "no_inferCNV_call": "#DDDDDD"}), lw=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title("Malignant cell calls", fontsize=8)
    panel_label(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    comp = pd.read_csv(os.path.join(lib.RES, "sc", "celltype_composition.csv"),
                       index_col=0)
    frac = comp.div(comp.sum(axis=1), axis=0)
    keep = frac.mean().sort_values(ascending=False).head(9).index
    ax.boxplot([frac[c].dropna().values for c in keep], widths=0.55,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=PAL["green"], alpha=0.6, lw=0.7))
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels(keep, rotation=40, ha="right", fontsize=5.4)
    ax.set_ylabel("Fraction per donor", fontsize=7.5)
    ax.set_title("Cell-type composition", fontsize=8)
    panel_label(ax, "c")
    save_fig(fig, FIGD, "FigS12")


def _km_panel(ax, t, e, g, colors, title):
    t = np.asarray(t, float); e = np.asarray(e, float); g = np.asarray(g)
    keep = ~(np.isnan(t) | np.isnan(e)) & (t > 0)
    t, e, g = t[keep], e[keep], g[keep]
    km = lib.km_curve(t, e, g)
    for name, (xs, ys, n) in km.items():
        ax.step(xs, ys, where="post", lw=1.1, color=colors.get(name, PAL["grey"]),
                label=f"{name} (n={n})")
    p = lib.logrank(t, e, g)[1]
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, np.nanmax(t) * 1.02)
    ax.set_xlabel("Months", fontsize=7)
    ax.set_ylabel("Survival probability", fontsize=7)
    ax.set_title(f"{title}\nlog-rank p = {p:.3g}", fontsize=7.4)
    ax.legend(fontsize=5.4, loc="upper right")
    return p, int(len(t)), int(np.nansum(e))


def S4():
    """Subtype survival across every annotated clinical endpoint.

    TCGA-BRCA has far fewer deaths than progression/recurrence events, so the
    four endpoints are reported side by side instead of relying on OS alone.
    """
    epi = [("OS", "os_time", "os_event", "Overall survival"),
           ("PFS", "pfs_time", "pfs_event", "Progression-free survival"),
           ("DSS", "dss_time", "dss_event", "Disease-specific survival"),
           ("DFS", "dfs_time", "dfs_event", "Disease-free survival")]
    fig = plt.figure(figsize=(9.6, 5.2))
    gs = GridSpec(2, 4, figure=fig, hspace=0.75, wspace=0.45,
                  height_ratios=[1, 0.85])
    rows = []
    for j, (tag, tc, ec, title) in enumerate(epi):
        ax = fig.add_subplot(gs[0, j])
        p, n, nev = _km_panel(ax, SUB[tc], SUB[ec], SUB["cluster"], NCOL, title)
        panel_label(ax, "abcd"[j], x=-0.22)
        rows.append({"endpoint": tag, "n": n, "events": nev, "logrank_p": p})

        # univariable HR for the second subtype against the first
        from lifelines import CoxPHFitter
        d = SUB[[tc, ec, "cluster"]].dropna()
        d = d[d[tc] > 0]
        d["grp"] = (d["cluster"] == SUBS[-1]).astype(int)
        hr = lo = hi = hp = np.nan
        if d["grp"].sum() >= 5 and d["grp"].nunique() > 1:
            cph = CoxPHFitter()
            cph.fit(d[["grp", tc, ec]], duration_col=tc, event_col=ec)
            s = cph.summary.loc["grp"]
            hr, lo, hi, hp = (s["exp(coef)"], s["exp(coef) lower 95%"],
                              s["exp(coef) upper 95%"], s["p"])
        rows[-1].update({"HR": hr, "lower": lo, "upper": hi, "HR_p": hp})

    ax = fig.add_subplot(gs[1, :])
    y = np.arange(len(rows))[::-1]
    for i, r in zip(y, rows):
        ax.plot([r["lower"], r["upper"]], [i, i], color=PAL["red"], lw=1.0)
        ax.plot(r["HR"], i, "o", color=PAL["red"], ms=3.4)
        ax.text(r["upper"] + 0.1, i,
                f"{r['HR']:.2f} ({r['lower']:.2f}-{r['upper']:.2f})  "
                f"p = {r['HR_p']:.3g}", va="center", fontsize=6)
    ax.axvline(1, color="#999999", lw=0.6, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['endpoint']}  (n={r['n']}, {r['events']} events)"
                        for r in rows], fontsize=6.6)
    ax.set_xlim(0, 5.4)
    ax.set_xlabel(f"Hazard ratio ({SUBS[-1]} vs {SUBS[0]})", fontsize=7.5)
    ax.set_title("Univariable Cox per endpoint", fontsize=8)
    panel_label(ax, "e", x=-0.16)
    pd.DataFrame(rows).to_csv(os.path.join(RESD, "subtype_survival_endpoints.csv"),
                              index=False)
    save_fig(fig, FIGD, "FigS4")


def S10():
    """Survival of the model-derived risk groups for OS and DFS."""
    rt = os.path.join(lib.RES, "model", "risk_table.csv")
    if not os.path.exists(rt):
        print("  (risk_table.csv missing - run fig4_5.py first)")
        return
    d = pd.read_csv(rt)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    cols = {"OS": ("os_time", "os_event"), "DFS": ("dfs_time", "dfs_event")}
    rows = []
    for k, (ax, (tag, (tc, ec))) in enumerate(zip(axes, cols.items())):
        if tc not in d:
            continue
        p, n, nev = _km_panel(ax, d[tc], d[ec], d["risk_group"],
                              {"High risk": PAL["red"], "Low risk": PAL["navy"]},
                              f"{tag} by risk group")
        panel_label(ax, "ab"[k])
        rows.append({"endpoint": tag, "n": n, "events": nev, "logrank_p": p})
    pd.DataFrame(rows).to_csv(os.path.join(RESD, "risk_survival_endpoints.csv"),
                              index=False)
    save_fig(fig, FIGD, "FigS10")


if __name__ == "__main__":
    which = sys.argv[1:] or [f"S{i}" for i in range(1, 13)]
    for w in which:
        try:
            globals()[w]()
            print(f"{w} done")
        except Exception as e:  # noqa: BLE001
            print(f"{w} FAILED: {e}")
    print("DONE")
