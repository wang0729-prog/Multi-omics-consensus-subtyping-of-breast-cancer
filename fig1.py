"""
fig1.py - Figure 1: multi-omics consensus clustering of TCGA-BRCA.

a  K-selection diagnostics (silhouette / PAC / mean algorithm agreement)
b  consensus matrix at the chosen K (with dendrogram-free ordering)
c  multi-omics feature heatmap ordered by subtype
d  Kaplan-Meier overall survival by subtype (TCGA-BRCA)
e  NTP projection onto METABRIC, agreement + KM
f  PAM50 vs subtype cross-tabulation (alluvial)

Outputs: figures/Fig1.*  +  results/fig1/*.csv (raw values for re-plotting)
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from style import (SUBTYPE_COLORS, apply_theme, panel_label, pval_text,
                   save_fig, PAL)

FIGD = os.path.join(lib.FIG)
RESD = os.path.join(lib.RES, "fig1")
os.makedirs(RESD, exist_ok=True)
apply_theme()

CL = lib.load_subtypes()
SUB = lib.merged_table()
SUBS = [c for c in CL["cluster"].unique()]
SUBS.sort()
NCOL = {s: SUBTYPE_COLORS.get(s, PAL["grey"]) for s in SUBS}


# ---- nearest-template prediction lives in lib.py (lib.ntp_templates /
# lib.ntp_predict) so that Fig1, Fig6 and the single-cell stage all use the
# same, correctly-oriented implementation.


# ---------------------------------------------------------------- KM
def plot_km(ax, time, event, group, colors, title="", tmax=None, show_n=True):
    t = np.asarray(time, float)
    e = np.asarray(event, float)
    g = np.asarray(group)
    keep = ~(np.isnan(t) | np.isnan(e)) & (t > 0)
    t, e, g = t[keep], e[keep], g[keep]
    km = lib.km_curve(t, e, g)
    for gi, (name, (xs, ys, n)) in enumerate(km.items()):
        col = colors.get(name, PAL["grey"]) if isinstance(colors, dict) else colors[gi]
        ax.step(xs, ys, where="post", color=col, lw=1.2, label=f"{name} (n={n})")
    chi2, p = lib.logrank(t, e, g)
    ax.set_xlabel("Time (months)", fontsize=7.5)
    ax.set_ylabel("Survival probability", fontsize=7.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, tmax or np.nanmax(t) * 1.02)
    ax.text(0.03, 0.06, f"log-rank p = {p:.1e}" if p < 0.01 else f"log-rank p = {p:.3f}",
            transform=ax.transAxes, fontsize=7)
    ax.legend(loc="upper right", fontsize=6.5)
    if title:
        ax.set_title(title, fontsize=8)
    return p


# ================================================================ build figure
def main():
    out = {}
    ks = pd.read_csv(os.path.join(lib.RES, "cluster", "k_selection.csv"))
    sub = pd.read_csv(os.path.join(lib.RES, "cluster", "subtypes.csv"))
    with open(os.path.join(lib.RES, "cluster", "feature_layers.pkl"), "rb") as f:
        fl = pickle.load(f)
    layers, samples = fl["layers"], fl["samples"]
    K = len(SUBS)
    C = pd.read_csv(os.path.join(lib.RES, "cluster", f"consensus_K{K}.csv"),
                    header=None).values

    fig = plt.figure(figsize=(9.2, 8.4))
    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42,
                  height_ratios=[1, 1, 1])

    # ---- a: K diagnostics
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ks))
    ax.bar(x - 0.18, ks["silhouette"], width=0.34, color=PAL["navy"],
           edgecolor="none", label="Silhouette")
    ax.bar(x + 0.18, ks["mean_within_algo_agreement"], width=0.34,
           color=PAL["red"], edgecolor="none", label="Algorithm agreement")
    ax2 = ax.twinx()
    ax2.plot(x, ks["PAC"], "-o", color=PAL["dark"], ms=2.6, lw=1.0, label="PAC")
    ax2.set_ylabel("PAC", fontsize=7.5)
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks["K"]])
    ax.set_ylabel("Score", fontsize=7.5)
    ax.set_title("Cluster-number selection", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    ax.axvline(list(ks["K"]).index(K), color=PAL["teal"], ls="--", lw=0.8)
    panel_label(ax, "a")

    # ---- b: consensus matrix
    ax = fig.add_subplot(gs[0, 1:])
    order = np.argsort(sub["cluster"].values)
    Co = C[np.ix_(order, order)]
    im = ax.imshow(Co, cmap="Blues", vmin=0, vmax=1, aspect="auto",
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Consensus", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    # subtype boundary lines + labels
    labs = sub["cluster"].values[order]
    bounds = np.where(labs[1:] != labs[:-1])[0] + 1
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.0)
        ax.axvline(b - 0.5, color="white", lw=1.0)
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(labs)]])
    for s, e in zip(starts, ends):
        ax.text(-len(labs) * 0.012, (s + e) / 2, labs[s], fontsize=7,
                va="center", ha="right", color=NCOL.get(labs[s], "#333"),
                fontweight="bold", clip_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Consensus matrix (K={K})", fontsize=8)
    panel_label(ax, "b", x=-0.06)

    # ---- c: multi-omics heatmap
    ax = fig.add_subplot(gs[1, :])
    blocks = []
    for name, X in layers.items():
        Z = pd.DataFrame(X, index=samples)
        Z = Z.rank(axis=1, pct=True).iloc[order]      # robust per-feature scaling
        n_show = min(60, Z.shape[1])
        pick = np.argsort(Z.var(axis=0))[::-1][:n_show]
        blocks.append(Z.iloc[:, pick].T)
    H = np.vstack([b.values for b in blocks])
    ax.imshow(H, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
              interpolation="nearest")
    bounds_x = []
    y0 = 0
    for b in blocks:
        y0 += b.shape[0]
        bounds_x.append(y0)
    for y in bounds_x[:-1]:
        ax.axhline(y - 0.5, color="white", lw=0.8)
    names = list(layers)
    y0 = 0
    for name, b in zip(names, blocks):
        ax.text(-0.008, y0 + b.shape[0] / 2, name, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7)
        y0 += b.shape[0]
    ax.set_yticks([])
    ax.set_xticks([])
    ybase = H.shape[0] + 3          # annotation strip below the last omics block
    for s, e in zip(starts, ends):
        ax.plot([s, e], [ybase, ybase], color=NCOL.get(labs[s], "#333"), lw=4,
                solid_capstyle="butt", clip_on=False)
        ax.text((s + e) / 2, ybase + 4, f"{labs[s]} (n={e-s})", ha="center",
                va="top", fontsize=7, clip_on=False,
                color=NCOL.get(labs[s], "#333"), fontweight="bold")
    ax.set_ylim(H.shape[0] + 14, -0.5)
    ax.set_title("Multi-omics feature profiles by subtype", fontsize=8)
    panel_label(ax, "c", x=-0.075, y=1.02)
    out["heatmap_rows"] = pd.DataFrame(H, columns=[samples[i] for i in order])

    # ---- e: KM TCGA (cited after the PAM50 cross-tab, so it takes letter e)
    # TCGA-BRCA carries relatively few deaths, so disease-free survival is the
    # better-powered endpoint; overall survival is reported alongside it.
    ax = fig.add_subplot(gs[2, 1])
    p_dfs = plot_km(ax, SUB["dfs_time"], SUB["dfs_event"], SUB["cluster"], NCOL,
                    "TCGA-BRCA, disease-free survival")
    p_os = lib.logrank(SUB["os_time"], SUB["os_event"], SUB["cluster"])[1]
    p_pfs = lib.logrank(SUB["pfs_time"], SUB["pfs_event"], SUB["cluster"])[1]
    ax.text(0.04, 0.006, f"OS p = {p_os:.3f}   PFS p = {p_pfs:.3f}",
            transform=ax.transAxes, fontsize=6, color="#555555")
    panel_label(ax, "e")
    out["tcga_km_p"] = pd.DataFrame(
        {"endpoint": ["DFS", "OS", "PFS"], "p": [p_dfs, p_os, p_pfs]})

    # ---- d: PAM50 cross-tab (cited first among the bottom-row panels)
    ax = fig.add_subplot(gs[2, 0])
    ct = pd.crosstab(SUB["pam50"], SUB["cluster"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
    frac = ct.div(ct.sum(axis=1), axis=0)
    y = np.arange(len(frac))[::-1]
    left = np.zeros(len(frac))
    for s in SUBS:
        ax.barh(y, frac[s], left=left, color=NCOL[s], edgecolor="white",
                height=0.62)
        for i, v in enumerate(frac[s]):
            if v > 0.12:
                ax.text(left[i] + v / 2, y[i], f"{int(ct[s].iloc[i])}", ha="center",
                        va="center", fontsize=6, color="white")
        left = left + frac[s].values
    ax.set_yticks(y)
    ax.set_yticklabels(frac.index, fontsize=7)
    ax.set_xlabel("Fraction of PAM50 class", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_title("PAM50 composition of each subtype", fontsize=8, pad=13)
    ax.legend(SUBS, fontsize=6, ncol=len(SUBS), loc="lower center",
              bbox_to_anchor=(0.5, 1.005), frameon=False)
    lib.__dict__  # noop
    from scipy.stats import chi2_contingency
    chi2, pchi, _, _ = chi2_contingency(ct.values)
    ax.text(0.02, -0.30, f"$\\chi^2$ p = {pchi:.1e}", transform=ax.transAxes,
            fontsize=7)
    panel_label(ax, "d")
    ct.to_csv(os.path.join(RESD, "pam50_crosstab.csv"))

    # ---- f: METABRIC NTP
    # The per-subtype signature is consumed by Fig6 and by the single-cell
    # stage, so it is written unconditionally before the projection is tried.
    mrna = pd.read_csv(os.path.join(lib.PRC, "mrna.csv"), index_col=0)
    mrna = mrna[[s for s in SUB["sample"] if s in mrna.columns]]
    tmpl, sig = lib.ntp_templates(mrna, SUB["cluster"].values, 200)
    pd.DataFrame({k: pd.Series(v) for k, v in sig.items()}).to_csv(
        os.path.join(RESD, "subtype_signature.csv"), index=False)
    print(f"  signature: {[ (k, len(v)) for k, v in sig.items() ]}")

    ax = fig.add_subplot(gs[2, 2])
    met_path = os.path.join(lib.PRC, "metabric_expression_panel.csv")
    if os.path.exists(met_path):
        met = pd.read_csv(met_path, index_col=0)
        common = [g for g in mrna.index if g in set(met.index)]
        if len(common) >= 100:
            Cm = lib.ntp_predict(tmpl, met.loc[common])
            pred = [SUBS[i] for i in np.nanargmax(Cm, axis=1)]
            # nearest-template confidence = margin between the best and the
            # runner-up template (an absolute correlation is not comparable
            # across platforms, a margin is)
            srt = np.sort(np.nan_to_num(Cm), axis=1)
            margin = srt[:, -1] - srt[:, -2]
            metlab = pd.DataFrame({"sample": met.columns, "pred": pred,
                                   "confidence": margin})
            metlab["pass"] = margin > 0.05
            metlab.to_csv(os.path.join(RESD, "metabric_ntp.csv"), index=False)
            mcl = pd.read_csv(os.path.join(lib.PRC, "metabric_clinical_patient.csv"),
                              index_col=0)
            metlab["patient"] = metlab["sample"].str[:12]
            m2 = metlab.merge(mcl, left_on="patient", right_index=True, how="left")
            # relapse-free survival is the endpoint with most events in
            # METABRIC; fall back to OS where RFS is missing
            m2["rfs_time"] = pd.to_numeric(m2.get("RFS_MONTHS"), errors="coerce")
            m2["rfs_event"] = (m2.get("RFS_STATUS").astype(str)
                               .str.startswith("1").astype(float))
            m2["rfs_event"] = m2["rfs_event"].where(
                m2["rfs_time"].notna(), np.nan)
            m2["os_time"] = pd.to_numeric(m2.get("OS_MONTHS"), errors="coerce")
            m2["os_event"] = (m2.get("OS_STATUS").astype(str)
                              .str.extract(r"^(\d)")[0].astype(float))
            ok = m2["pass"].fillna(False).values
            p_met = plot_km(ax, m2.loc[ok, "rfs_time"], m2.loc[ok, "rfs_event"],
                            m2.loc[ok, "pred"], NCOL,
                            "METABRIC (NTP-labelled), relapse-free survival")
            print(f"  METABRIC NTP: {int(ok.sum())}/{len(m2)} pass the margin rule")
            out["metabric_ntp"] = metlab
            out["metabric_km_p"] = pd.DataFrame({"p": [p_met]})
        else:
            ax.text(0.5, 0.5, f"only {len(common)} shared signature genes\n(METABRIC panel incomplete)",
                    ha="center", fontsize=7)
            ax.axis("off")
    else:
        ax.text(0.5, 0.5, "METABRIC panel not fetched", ha="center", fontsize=7)
        ax.axis("off")
    panel_label(ax, "f")

    fig.suptitle("")
    paths = save_fig(fig, FIGD, "Fig1")
    print("saved:", paths)
    for k, v in out.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(RESD, f"{k}.csv"), index=False)
    print("DONE")


if __name__ == "__main__":
    main()
