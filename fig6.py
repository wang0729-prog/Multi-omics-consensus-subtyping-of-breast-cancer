"""
fig6.py - Figure 6: CPTAC-BRCA proteomic / phosphoproteomic validation layer.

a  mRNA-protein correlation for the subtype signature genes
b  differential protein abundance (contrast subtypes projected onto CPTAC)
c  differential phosphorylation (site level)
d  top phosphosites on kinases (lollipop)
e  protein-level directional consistency of the subtype signature
f  proteomic risk score vs subtype

Outputs: figures/Fig6.*  and results/fig6/*.csv
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
from style import PAL, SUBTYPE_COLORS, apply_theme, panel_label, pval_text, save_fig

FIGD = lib.FIG
RESD = os.path.join(lib.RES, "fig6")
os.makedirs(RESD, exist_ok=True)
apply_theme()

SUBS = list(lib.contrast_pair())
C1, C2 = SUBS
NCOL = {s: SUBTYPE_COLORS.get(s, PAL["grey"]) for s in SUBS}


def read_cct(path):
    d = pd.read_csv(path, sep="\t", index_col=0)
    d.columns = [c for c in d.columns]
    return d


def volcano(ax, dep, xcol="log2FC", ycol="FDR", xlab="log$_2$ fold change",
            fc=0.5, fdr=0.05, nlab=10):
    dep = dep.copy()
    dep["nlp"] = -np.log10(dep[ycol].clip(lower=1e-300))
    sig = (dep[ycol] < fdr) & (dep[xcol].abs() > fc)
    ax.scatter(dep.loc[~sig, xcol], dep.loc[~sig, "nlp"], s=1.4, color="#D9D9D9",
               lw=0, rasterized=True)
    ax.scatter(dep.loc[sig & (dep[xcol] > 0), xcol],
               dep.loc[sig & (dep[xcol] > 0), "nlp"], s=1.8, color=PAL["red"], lw=0)
    ax.scatter(dep.loc[sig & (dep[xcol] < 0), xcol],
               dep.loc[sig & (dep[xcol] < 0), "nlp"], s=1.8, color=PAL["navy"], lw=0)
    tp = dep[sig].sort_values("nlp", ascending=False).head(nlab)
    for _, r in tp.iterrows():
        ax.annotate(str(r["gene"]), (r[xcol], r["nlp"]), fontsize=5.4,
                    xytext=(2, 1), textcoords="offset points")
    ax.axvline(0, color="#CCCCCC", lw=0.5)
    ax.axhline(-np.log10(fdr), color="#CCCCCC", lw=0.5, ls="--")
    ax.set_xlabel(xlab, fontsize=7.5)
    ax.set_ylabel("$-\\log_{10}$ FDR", fontsize=7.5)
    return dep


def main():
    raw = lib.RAW
    prot = read_cct(os.path.join(raw, "HS_CPTAC_BRCA_2018_Proteome_Ratio_Norm_gene_Median.cct"))
    phos = read_cct(os.path.join(raw, "HS_CPTAC_BRCA_2018_Phosphoproteome_Ratio_Norm_Site.cct"))
    pgen = read_cct(os.path.join(raw, "HS_CPTAC_BRCA_2018_Phosphoproteome_Ratio_Norm_Gene_median.cct"))
    rna = read_cct(os.path.join(raw, "HS_CPTAC_BRCA_2018_RNA_GENE.cct"))
    cli = pd.read_csv(os.path.join(raw, "HS_CPTAC_BRCA_2018_CLI.tsi"), sep="\t",
                      index_col=0)
    print(f"  proteome {prot.shape}, phosphosite {phos.shape}, rna {rna.shape}")

    samples = [c for c in prot.columns if c in rna.columns]
    rna = rna[samples]

    # ---- project TCGA subtypes onto CPTAC via NTP
    # templates are built from TCGA (subtype vs rest t-statistics); each
    # CPTAC sample is then scored against the TCGA centroids.  Building the
    # templates from CPTAC's own cohort mean (as before) makes both templates
    # nearly identical and collapses every sample into one subtype.
    mrna = pd.read_csv(os.path.join(lib.PRC, "mrna.csv"), index_col=0)
    sub = lib.load_subtypes()
    keep = [s for s in sub["sample"] if s in mrna.columns]
    groups = sub.set_index("sample").loc[keep, "cluster"].values
    templates, gene_lists = lib.ntp_templates(mrna[keep], groups, n_genes=200)
    C = lib.ntp_predict(templates, rna)          # samples x subtypes
    tnames = np.array(list(templates))
    ordr = np.argsort(-C, axis=1)
    ri = np.arange(C.shape[0])
    lab = tnames[ordr[:, 0]]
    conf = C[ri, ordr[:, 0]] - C[ri, ordr[:, 1]]   # margin = confidence
    cptac_lab = pd.DataFrame({"sample": rna.columns, "subtype": lab,
                              "margin": conf})
    cptac_lab.to_csv(os.path.join(RESD, "cptac_ntp_labels.csv"), index=False)
    print(cptac_lab["subtype"].value_counts().to_dict())
    sigsets = {c: gene_lists[c] for c in templates}

    labmap = dict(zip(cptac_lab["sample"], cptac_lab["subtype"]))
    common = [s for s in samples if s in labmap]
    g1 = [s for s in common if labmap[s] == C1]
    g2 = [s for s in common if labmap[s] == C2]
    print(f"  CPTAC {C1}={len(g1)} {C2}={len(g2)}")

    # ---- differential proteins
    def diff(mat, a, b):
        A, B = mat[a], mat[b]
        mn = A.mean(axis=1) - B.mean(axis=1)
        rows = []
        for i, g in enumerate(mat.index):
            x, y = A.iloc[i].values, B.iloc[i].values
            x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
            if len(x) < 5 or len(y) < 5:
                continue
            t, p = stats.ttest_ind(x, y, equal_var=False)
            rows.append((g, mn.iloc[i], p, len(x) + len(y)))
        d = pd.DataFrame(rows, columns=["gene", "log2FC", "p", "n"])
        d["FDR"] = lib.bh(d["p"].fillna(1))
        return d.dropna(subset=["log2FC"])

    dprot = diff(prot, g1, g2)
    dprot.to_csv(os.path.join(RESD, "differential_proteins.csv"), index=False)
    dphos = diff(phos, g1, g2)
    dphos["site"] = [str(g).split("|")[0] if "|" in str(g) else str(g)
                     for g in dphos["gene"]]
    dphos.to_csv(os.path.join(RESD, "differential_phosphosites.csv"), index=False)
    dpg = diff(pgen, g1, g2)
    dpg.to_csv(os.path.join(RESD, "differential_phospho_genes.csv"), index=False)
    print(f"  proteins sig: {(dprot['FDR']<0.05).sum()}, "
          f"phosphosites sig: {(dphos['FDR']<0.05).sum()}")

    # ---- mRNA-protein correlation for signature genes
    genes_shared = [g for g in set(prot.index) & set(rna.index)]
    gs = list(dict.fromkeys([g for c in SUBS for g in sigsets[c]]))
    gs = [g for g in gs if g in genes_shared]
    rows = []
    for g in gs:
        x = rna.loc[g, common]; y = prot.loc[g, common]
        ok = x.notna() & y.notna()
        if ok.sum() < 20:
            continue
        r, p = stats.spearmanr(x[ok], y[ok])
        rows.append({"gene": g, "rho": r, "p": p, "n": int(ok.sum())})
    cor = pd.DataFrame(rows).dropna()
    cor.to_csv(os.path.join(RESD, "mrna_protein_correlation.csv"), index=False)

    # ---- kinases among differential phospho genes
    hgnc = pd.read_csv(os.path.join(lib.RAW, "hgnc_complete_set.txt"), sep="\t",
                       low_memory=False)
    kin = set(hgnc.loc[hgnc["gene_group"].astype(str).str.contains("inase",
                 case=False, na=False), "symbol"].astype(str))

    # ================================================= figure
    fig = plt.figure(figsize=(9.2, 6.4))
    gs2 = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.55)

    ax = fig.add_subplot(gs2[0, 0])
    if len(cor):
        ax.scatter(cor["rho"], -np.log10(cor["p"].clip(lower=1e-300)), s=3,
                   color=PAL["teal"], alpha=0.7, lw=0)
        ax.axvline(0, color="#CCCCCC", lw=0.5)
        frac = (cor["rho"] > 0.3).mean()
        ax.text(0.03, 0.95, f"{frac*100:.0f}% of signature genes\nwith rho>0.3",
                transform=ax.transAxes, fontsize=6, va="top")
        ax.set_xlabel("Spearman rho (mRNA vs protein)", fontsize=7.5)
        ax.set_ylabel("$-\\log_{10}$ p", fontsize=7.5)
        ax.set_title("mRNA-protein concordance", fontsize=8)
    else:
        ax.axis("off")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs2[0, 1])
    volcano(ax, dprot, fc=0.3, nlab=6)
    ax.set_title("Differential proteins", fontsize=8)
    ax.set_xlabel(f"log$_2$ FC protein ({C1}/{C2})", fontsize=7.5)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs2[0, 2])
    volcano(ax, dphos, fc=0.4, nlab=6)
    ax.set_title("Differential phosphosites", fontsize=8)
    ax.set_xlabel("log$_2$ FC phosphosite", fontsize=7.5)
    panel_label(ax, "c")

    # d kinase lollipop
    ax = fig.add_subplot(gs2[1, 0])
    dk = dpg[dpg["gene"].astype(str).isin(kin)].copy()
    dk = dk.reindex(dk["log2FC"].abs().sort_values(ascending=False).index).head(14)
    dk = dk.sort_values("log2FC")
    if len(dk):
        for i, (_, r) in enumerate(dk.iterrows()):
            c = PAL["red"] if r["log2FC"] > 0 else PAL["navy"]
            ax.plot([0, r["log2FC"]], [i, i], color=c, lw=1.0)
            ax.plot(r["log2FC"], i, "o", color=c, ms=3.2)
        ax.set_yticks(range(len(dk)))
        ax.set_yticklabels(dk["gene"], fontsize=6)
        ax.axvline(0, color="#999999", lw=0.6)
        ax.set_xlabel(f"log$_2$ FC phospho ({C1}/{C2})", fontsize=7.5)
        ax.set_title("Kinase phosphorylation", fontsize=8)
    panel_label(ax, "d")

    # e directional consistency heatmap
    ax = fig.add_subplot(gs2[1, 1])
    dm = dprot.set_index("gene")["log2FC"]
    vals = []
    for c in SUBS:
        v = [dm.get(g, np.nan) for g in sigsets[c]]
        vals.append(v)
    allv = np.concatenate(vals)
    allv = allv[~np.isnan(allv)]
    if len(allv):
        bins = np.linspace(np.nanpercentile(allv, 2), np.nanpercentile(allv, 98), 31)
        for j, c in enumerate(SUBS):
            v = np.array(vals[j], float)
            v = v[~np.isnan(v)]
            h, _ = np.histogram(v, bins=bins, density=True)
            ax.plot((bins[:-1] + bins[1:]) / 2, h, color=NCOL[c], lw=1.1, label=c)
        ax.axvline(0, color="#999999", lw=0.6, ls="--")
        ax.set_xlabel(f"Protein log$_2$ FC ({C1}/{C2})", fontsize=7.5)
        ax.set_ylabel("Density", fontsize=7.5)
        ax.set_title("Protein-level direction of the signature", fontsize=8)
        ax.legend(fontsize=6)
    panel_label(ax, "e")

    # f proteomic score - signed, direction-aware subtype score.  The naive
    # "mean of the top 40 differential proteins" cancelled the contrast because
    # the table is alphabetically ordered and mixed BC1- and BC2-high proteins;
    # select the top 20 of each direction and score BC1-high minus BC2-high.
    ax = fig.add_subplot(gs2[1, 2])
    sig = dprot[dprot["FDR"] < 0.05]
    up = sig.nlargest(20, "log2FC")["gene"].tolist()    # BC1-enriched
    dn = sig.nsmallest(20, "log2FC")["gene"].tolist()   # BC2-enriched
    panel = [g for g in up + dn if g in prot.index]
    if len(panel) >= 6:
        P = prot.loc[panel, common]
        Z = P.sub(P.mean(axis=1), axis=0).div(
            P.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        sign = pd.Series(1.0, index=panel)
        sign.loc[[g for g in dn if g in prot.index]] = -1.0
        score = Z.mul(sign, axis=0).mean(axis=0)
        for j, c in enumerate(SUBS):
            v = score[[s for s in common if labmap[s] == c]].dropna()
            ax.scatter(j + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)),
                       v, s=4, color=NCOL[c], alpha=0.7, lw=0)
            ax.plot([j - 0.3, j + 0.3], [v.median()] * 2, color="#1B1B1B", lw=1.0)
        pv = lib.mannwhitney(*[score[[s for s in common if labmap[s] == c]].values
                               for c in SUBS])
        ax.set_xticks([0, 1]); ax.set_xticklabels(SUBS, fontsize=7)
        ax.set_ylabel("Proteomic subtype score", fontsize=7.5)
        ax.set_title(f"Proteomic subtype score ({pval_text(pv)})", fontsize=8)
    panel_label(ax, "f")

    save_fig(fig, FIGD, "Fig6")
    print("DONE")


if __name__ == "__main__":
    main()
