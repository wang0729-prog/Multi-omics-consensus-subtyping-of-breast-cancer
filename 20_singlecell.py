"""
20_singlecell.py - Figures 7 & 8: single-cell landscape and cell-cell
communication of breast cancer (Wu et al. 2021 atlas, CELLxGENE, 100,064 cells).

Fig7 a UMAP by major cell type
     b cell-type composition per donor (stacked)
     c malignant vs normal epithelium (inferCNV calls)
     d subtype signature score on malignant cells (UMAP)
     e malignant-cell subtype assignment (contrast subtypes)
     f Hallmark pathway activity in malignant cells
     g metabolic programme score
Fig8 a ligand-receptor interaction network across cell types (circle)
     b outgoing / incoming signalling strength
     c key ligand-receptor pairs by malignant subtype (heatmap)
     d MIF / CXCL / MHC axis summary

Outputs: figures/Fig7.*, figures/Fig8.*  +  results/sc/*.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from style import PAL, SUBTYPE_COLORS, apply_theme, panel_label, pval_text, save_fig

FIGD = lib.FIG
RESD = os.path.join(lib.RES, "sc")
os.makedirs(RESD, exist_ok=True)
apply_theme()

C1, C2 = lib.contrast_pair()

H5 = os.path.join(lib.RAW, "singlecell", "brca_wu2021_atlas.h5ad")
CACHE = os.path.join(RESD, "sc_processed.h5ad")


def build_adata():
    import anndata as ad
    import scanpy as sc
    if os.path.exists(CACHE):
        a = ad.read_h5ad(CACHE)
        if a.raw is not None:
            # stale raw slot keeps the Ensembl var_names and silently diverts
            # scanpy scoring (use_raw=None -> raw) to gene IDs
            del a.raw
        return a
    a = ad.read_h5ad(H5)
    # CELLxGENE stores Ensembl IDs as var_names; the HGNC symbols live in
    # var["feature_name"] - every signature / resource lookup needs symbols
    if "feature_name" in a.var.columns:
        # plain str list (object dtype): a pandas StringDtype index breaks
        # Index.intersection in scanpy's score_genes on this pandas version
        a.var_names = [str(x) for x in a.var["feature_name"]]
    a.var_names_make_unique()
    if a.raw is not None:
        del a.raw
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    # the embedding is computed on highly variable genes only (scaling the full
    # 100,064 x 28,468 matrix would densify to ~11 GB); gene-set scoring below
    # still uses the complete gene space, and the UMAP/leiden results are
    # transferred back onto the full object.
    hv = sc.pp.highly_variable_genes(a, n_top_genes=2500, batch_key="donor_id",
                                     inplace=False)
    emb = a[:, hv["highly_variable"].values].copy()
    sc.pp.scale(emb, max_value=10)
    sc.tl.pca(emb, n_comps=40, svd_solver="randomized")
    sc.pp.neighbors(emb, n_neighbors=15, n_pcs=30)
    sc.tl.umap(emb)
    sc.tl.leiden(emb, resolution=1.0, key_added="leiden", flavor="igraph",
                 n_iterations=2, directed=False)
    a.obsm["X_umap"] = emb.obsm["X_umap"]
    a.obs["leiden"] = emb.obs["leiden"].values
    a.write_h5ad(CACHE)
    return a


def score_sets(a, sets, prefix):
    import scanpy as sc
    for k, v in sets.items():
        v = [g for g in v if g in a.var_names]
        if len(v) >= 5:
            sc.tl.score_genes(a, v, score_name=f"{prefix}_{k}")
    return a


def main():
    import scanpy as sc
    a = build_adata()
    print(f"  loaded {a.shape}")

    MAJ = "celltype_major"
    SUB = "subtype"          # TNBC / ER+ / HER2+
    CALL = "normal_cell_call"
    major = list(a.obs[MAJ].astype(str).unique())
    mcol = dict(zip(major,
            [PAL["red"], PAL["navy"], PAL["teal"], PAL["salmon"], PAL["purple"],
             PAL["orange"], PAL["slate"], PAL["green"], PAL["brown"],
             PAL["pink"], PAL["grey"], PAL["blue"]][:len(major)]))

    # ---- malignant cells and subtype score
    mal = a.obs[CALL].astype(str).isin(["cancer"])
    sigp = os.path.join(lib.RES, "fig1", "subtype_signature.csv")
    sub_genes = {}
    if os.path.exists(sigp):
        s = pd.read_csv(sigp)
        for c in s.columns:
            sub_genes[c] = [g for g in s[c].dropna().tolist() if g in a.var_names]
    if not sub_genes:
        dep = pd.read_csv(os.path.join(lib.RES, "fig2", "DEG_contrast.csv"))
        sub_genes = {C1: [g for g in dep.sort_values("log2FC", ascending=False)
                             ["gene"].head(200) if g in a.var_names],
                     C2: [g for g in dep.sort_values("log2FC")["gene"].head(200)
                             if g in a.var_names]}
    for c, g in sub_genes.items():
        if len(g) >= 5:
            sc.tl.score_genes(a, g, score_name=f"sig_{c}")

    # malignant cells only
    am = a[mal].copy()
    for c in sub_genes:
        col = f"sig_{c}"
        if col in am.obs:
            pass
    score_cols = [c for c in am.obs.columns if c.startswith("sig_")]
    if len(score_cols) >= 2:
        M = am.obs[score_cols].values
        # z-score across malignant cells then assign
        Z = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
        am.obs["sc_subtype"] = [score_cols[i].replace("sig_", "")
                                for i in np.argmax(Z, axis=1)]
        am.obs["sc_conf"] = Z.max(axis=1)
    else:
        am.obs["sc_subtype"] = "NA"
        am.obs["sc_conf"] = 0

    # ---- pathway scores in malignant cells
    hall = lib.load_gene_sets("h.all")
    keep_sets = lib.hallmark_pick(hall, {
        "E2F Targets": ("e2f",),
        "G2M Checkpoint": ("g2m",),
        "EMT": ("epithelial", "mesenchymal"),
        "Hypoxia": ("hypoxia",),
        "Inflammatory Response": ("inflammatory",),
        "IL6-JAK-STAT3": ("il6",),
        "TNF-alpha via NF-kB": ("tnf",),
        "PI3K-AKT-mTOR": ("pi3k",),
        "OXPHOS": ("oxidative phosphorylation",),
        "Glycolysis": ("glycolysis",)})
    score_sets(am, keep_sets, "pw")

    # ---- metabolic score
    metab = lib.hallmark_pick(hall, {
        "Glycolysis": ("glycolysis",),
        "OXPHOS": ("oxidative phosphorylation",),
        "Fatty Acid": ("fatty acid",)})
    score_sets(am, metab, "mb")

    # ---- save tables
    um = pd.DataFrame(a.obsm["X_umap"], columns=["UMAP1", "UMAP2"],
                      index=a.obs_names)
    um[MAJ] = a.obs[MAJ].astype(str).values
    um[SUB] = a.obs[SUB].astype(str).values
    um["donor"] = a.obs["donor_id"].astype(str).values
    um["call"] = a.obs[CALL].astype(str).values
    um.to_csv(os.path.join(RESD, "umap_all.csv"))

    amm = pd.DataFrame(am.obsm["X_umap"], columns=["UMAP1", "UMAP2"],
                       index=am.obs_names)
    amm["sc_subtype"] = am.obs["sc_subtype"].values
    amm["donor"] = am.obs["donor_id"].astype(str).values
    amm.to_csv(os.path.join(RESD, "umap_malignant.csv"))

    comp = pd.crosstab(a.obs["donor_id"].astype(str), a.obs[MAJ].astype(str))
    comp.to_csv(os.path.join(RESD, "celltype_composition.csv"))

    # ---- pathway summary by malignant subtype
    pwcols = [c for c in am.obs.columns if c.startswith("pw_")]
    mbcols = [c for c in am.obs.columns if c.startswith("mb_")]
    rows = []
    for c in pwcols + mbcols:
        v1 = am.obs.loc[am.obs["sc_subtype"] == C1, c].values
        v2 = am.obs.loc[am.obs["sc_subtype"] == C2, c].values
        if len(v1) > 20 and len(v2) > 20:
            rows.append({"programme": c, C1: np.mean(v1), C2: np.mean(v2),
                         "p": lib.mannwhitney(v1, v2)})
    pws = pd.DataFrame(rows)
    pws.to_csv(os.path.join(RESD, "pathway_by_sc_subtype.csv"), index=False)

    # ---- liana cell-cell communication
    lr = None
    try:
        import liana as li
        a2 = a.copy()
        # liana permutes cells, so cost grows quickly with n_obs; keep every
        # cell type but cap the total at ~30k cells (stratified)
        if a2.n_obs > 30000:
            rng = np.random.default_rng(20260911)
            lab = a2.obs[MAJ].astype(str).values
            keep = []
            for m in np.unique(lab):
                idx = np.where(lab == m)[0]
                n = int(min(len(idx), max(500, 30000 * len(idx) / a2.n_obs)))
                keep.append(rng.choice(idx, n, replace=False))
            keep = np.sort(np.concatenate(keep))
            a2 = a2[keep].copy()
            print(f"  liana on subsample {a2.shape}", flush=True)
        li.method.rank_aggregate(a2, groupby=MAJ, resource_name="consensus",
                                 expr_prop=0.1, verbose=False,
                                 use_raw=False, n_perms=50,
                                 key_added="liana_res")
        lr = a2.uns["liana_res"]
        lr.to_csv(os.path.join(RESD, "liana_all.csv"), index=False)
        print(f"  liana: {lr.shape}")
    except Exception as e:  # noqa: BLE001
        print("  liana failed:", e)

    # ================================================= FIGURE 7
    fig = plt.figure(figsize=(9.4, 8.0))
    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    for m in major:
        d = um[um[MAJ] == m]
        ax.scatter(d["UMAP1"], d["UMAP2"], s=0.12, color=mcol[m], lw=0,
                   rasterized=True, label=m)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title(f"UMAP, {a.n_obs:,} cells", fontsize=8)
    ax.legend(fontsize=4.6, markerscale=8, loc="center left",
              bbox_to_anchor=(1.0, 0.5), frameon=False)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for s in ["TNBC", "ER+", "HER2+"]:
        d = um[um[SUB] == s]
        ax.scatter(d["UMAP1"], d["UMAP2"], s=0.1, lw=0, rasterized=True,
                   color={"TNBC": PAL["red"], "ER+": PAL["navy"],
                          "HER2+": PAL["teal"]}.get(s, PAL["grey"]), label=s)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title("Clinical subtype", fontsize=8)
    ax.legend(fontsize=5.4, markerscale=8, frameon=False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    for c, col in [("cancer", PAL["red"]), ("normal", PAL["navy"]),
                   ("no_inferCNV_call", "#CCCCCC")]:
        d = um[um["call"] == c]
        ax.scatter(d["UMAP1"], d["UMAP2"], s=0.1, lw=0, rasterized=True,
                   color=col, label=f"{c} ({len(d):,})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title("Malignant vs normal (inferCNV)", fontsize=8)
    ax.legend(fontsize=5.4, markerscale=8, frameon=False)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    if f"sig_{C1}" in am.obs:
        sc1 = am.obs[f"sig_{C1}"].values - am.obs[f"sig_{C2}"].values
        o = np.argsort(sc1)
        sct = ax.scatter(amm["UMAP1"].values[o], amm["UMAP2"].values[o], s=0.2,
                         c=sc1[o], cmap="RdBu_r", vmin=-np.percentile(np.abs(sc1), 98),
                         vmax=np.percentile(np.abs(sc1), 98), lw=0, rasterized=True)
        cb = fig.colorbar(sct, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"sig({C1}) - sig({C2})", fontsize=6)
        cb.ax.tick_params(labelsize=5.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title("Subtype signature on malignant cells", fontsize=8)
    panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    for s in (C1, C2):
        d = amm[amm["sc_subtype"] == s]
        ax.scatter(d["UMAP1"], d["UMAP2"], s=0.2, lw=0, rasterized=True,
                   color=SUBTYPE_COLORS[s], label=f"{s} (n={len(d):,})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=7); ax.set_ylabel("UMAP2", fontsize=7)
    ax.set_title("Malignant-cell subtype assignment", fontsize=8)
    ax.legend(fontsize=5.6, markerscale=8, frameon=False)
    panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    mal_sub = pd.crosstab(am.obs["donor_id"].astype(str), am.obs["sc_subtype"])
    frac = mal_sub.div(mal_sub.sum(axis=1), axis=0).fillna(0)
    frac = frac.loc[mal_sub.sum(axis=1).sort_values(ascending=False).index]
    y = np.arange(len(frac))[::-1]
    left = np.zeros(len(frac))
    for s in (C1, C2):
        if s in frac:
            ax.barh(y, frac[s], left=left, color=SUBTYPE_COLORS[s], height=0.7)
            left = left + frac[s].values
    ax.set_yticks(y); ax.set_yticklabels(frac.index, fontsize=4.8)
    ax.set_xlabel("Fraction of malignant cells", fontsize=7)
    ax.set_title("Malignant subtype per donor", fontsize=8, pad=14)
    ax.legend([C1, C2], fontsize=5.6, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False)
    panel_label(ax, "f", x=-0.22)

    ax = fig.add_subplot(gs[2, :2])
    if len(pws):
        p = pws.copy()
        p["name"] = p["programme"].str.replace("pw_", "").str.replace("mb_", "")
        p = p.sort_values(C1)
        x = np.arange(len(p))
        ax.barh(x - 0.19, p[C1], height=0.36, color=SUBTYPE_COLORS[C1])
        ax.barh(x + 0.19, p[C2], height=0.36, color=SUBTYPE_COLORS[C2])
        ax.set_yticks(x)
        ax.set_yticklabels([n.replace("_", " ").title() for n in p["name"]],
                           fontsize=5.8)
        for i, pv in enumerate(p["p"]):
            ax.text(max(p[C1].iloc[i], p[C2].iloc[i]) + 0.002, i,
                    pval_text(pv), fontsize=5, va="center")
        ax.set_xlabel("Mean score", fontsize=7.5)
        ax.set_title("Pathway and metabolic programmes in malignant cells", fontsize=8)
        ax.legend([C1, C2], fontsize=6, ncol=2)
    panel_label(ax, "g")

    ax = fig.add_subplot(gs[2, 2])
    ct = a.obs[[MAJ, CALL]].copy()
    ct = pd.crosstab(ct[MAJ].astype(str), ct[CALL].astype(str))
    ct = ct.div(ct.sum(axis=1), axis=0)
    calls = [c for c in ["cancer", "normal", "no_inferCNV_call"] if c in ct]
    colours = {"cancer": PAL["salmon"], "normal": PAL["navy"],
               "no_inferCNV_call": "#cccccc"}
    left = np.zeros(len(ct))
    ypos = np.arange(len(ct))
    for call in calls:
        vals = ct[call].values if call in ct else np.zeros(len(ct))
        ax.barh(ypos, vals, left=left, color=colours[call], height=0.7,
                label=call.replace("_", " "))
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(ct.index, fontsize=5.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of cells", fontsize=7)
    ax.set_title("inferCNV call composition by cell type", fontsize=8)
    ax.legend(fontsize=5, loc="lower right", framealpha=0.9)
    panel_label(ax, "h")
    save_fig(fig, FIGD, "Fig7")
    print("Fig7 done")

    # ================================================= FIGURE 8
    fig = plt.figure(figsize=(9.0, 6.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.55)

    if lr is not None and len(lr):
        lr = lr.copy()
        scols = [c for c in lr.columns]
        score_col = ("magnitude_rank" if "magnitude_rank" in scols else
                     "lr_means" if "lr_means" in scols else None)
        spec_col = "specificity" if "specificity" in scols else None
        if score_col:
            lr["w"] = (1 - lr[score_col]) if "rank" in score_col else lr[score_col]
        else:
            lr["w"] = 1.0

        ax = fig.add_subplot(gs[0, 0])
        net = lr.groupby(["source", "target"])["w"].mean().reset_index()
        net.to_csv(os.path.join(RESD, "lr_network.csv"), index=False)
        out = net.groupby("source")["w"].sum().sort_values(ascending=False)
        inn = net.groupby("target")["w"].sum().reindex(out.index).fillna(0)
        ax.scatter(out, inn, s=22, color=PAL["navy"], lw=0)
        for k in out.index:
            ax.annotate(k, (out[k], inn[k]), fontsize=5.2,
                        xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("Outgoing interaction strength", fontsize=7.5)
        ax.set_ylabel("Incoming interaction strength", fontsize=7.5)
        ax.set_title("Signalling role of each cell type", fontsize=8)
        panel_label(ax, "a")

        ax = fig.add_subplot(gs[0, 1])
        top = net.sort_values("w", ascending=False).head(18)
        top["pair"] = top["source"] + " -> " + top["target"]
        top = top.sort_values("w")
        ax.barh(np.arange(len(top)), top["w"], color=PAL["teal"], height=0.7)
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(top["pair"], fontsize=5.2)
        ax.set_xlabel("Mean interaction strength", fontsize=7.5)
        ax.set_title("Top cell-cell interactions", fontsize=8)
        panel_label(ax, "b")

        ax = fig.add_subplot(gs[0, 2])
        lrx = lr.copy()
        lrx["lr"] = lrx["ligand_complex"].astype(str) + "-" + \
                    lrx["receptor_complex"].astype(str)
        glob = lrx.groupby("lr")["w"].mean().sort_values(ascending=False)
        top_lr = glob.head(30).index
        sub = lrx[lrx["lr"].isin(top_lr)]
        piv = sub.pivot_table(index="lr", columns="source", values="w",
                              aggfunc="mean").reindex(top_lr)
        piv = piv.reindex(columns=[c for c in piv.columns])
        im = ax.imshow(piv.values, aspect="auto", cmap="magma")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=90, fontsize=4.6)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=4.4)
        ax.set_title("Top ligand-receptor pairs", fontsize=8)
        panel_label(ax, "c")

        ax = fig.add_subplot(gs[1, :])
        fam = lrx.copy()
        fam["lig"] = fam["ligand_complex"].astype(str)
        families = {"MIF": ["MIF", "CD74", "CXCR4", "CD44", "ACKR3"],
                    "CXCL": ["CXCL9", "CXCL10", "CXCL11", "CXCL12", "CXCL13",
                             "CXCR3", "CXCR4", "CXCR5", "ACKR3"],
                    "MHC": ["HLA-A", "HLA-B", "HLA-C", "HLA-DMA", "HLA-DMB",
                            "HLA-DRA", "CD74", "CD8A", "CD4"],
                    "TGFB": ["TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2"],
                    "WNT": ["WNT1", "WNT2", "WNT3", "WNT4", "WNT5A", "WNT5B",
                            "FZD1", "FZD2", "FZD3", "FZD4", "FZD5", "LRP5", "LRP6"],
                    "VEGF": ["VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "FLT4"],
                    "FN1": ["FN1", "ITGA5", "ITGB1", "ITGAV", "ITGB5", "SDC1"]}
        rows = []
        for famname, genes in families.items():
            sub = fam[fam["lig"].isin(genes)]
            if len(sub):
                s = sub.groupby("source")["w"].mean()
                rows.append({"family": famname, **{k: v for k, v in s.items()}})
        if rows:
            fdf = pd.DataFrame(rows).set_index("family")
            cols = [c for c in major if c in fdf.columns]
            fdf = fdf[cols]
            im = ax.imshow(fdf.values, aspect="auto", cmap="viridis")
            cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
            cb.ax.tick_params(labelsize=5.5)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=5.6)
            ax.set_yticks(range(len(fdf)))
            ax.set_yticklabels(fdf.index, fontsize=6.4)
            ax.set_title("Signalling families across cell types", fontsize=8)
            fdf.to_csv(os.path.join(RESD, "lr_families.csv"))
        panel_label(ax, "d")
    else:
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.5, 0.5, "ligand-receptor analysis unavailable", ha="center",
                fontsize=7)
        ax.axis("off")

    save_fig(fig, FIGD, "Fig8")
    print("Fig8 done")
    print("DONE")


if __name__ == "__main__":
    main()
