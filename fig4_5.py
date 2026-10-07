"""
fig4_5.py - Figures 4 & 5: machine-learning prognostic model.

Fig4  a  101-combination model performance (mean CV C-index)
      b  final model coefficients (forest plot)
      c  risk-score distribution by subtype
      d  risk-ordered triple panel (risk curve / survival status / gene heatmap)
      e  Kaplan-Meier high vs low risk (TCGA-BRCA)
      f  time-dependent ROC (1/3/5-year)
Fig5  a  benchmark of the model against clinical & molecular predictors
      b  nomogram integrating the risk score and clinical variables
      c  calibration curve (bootstrap)
      d  decision-curve analysis

Outputs: figures/Fig4.*, figures/Fig5.*, results/model/*.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from lifelines import CoxPHFitter
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import models as M
from style import PAL, SUBTYPE_COLORS, apply_theme, panel_label, pval_text, save_fig

FIGD = lib.FIG
RESD = os.path.join(lib.RES, "model")
os.makedirs(RESD, exist_ok=True)
apply_theme()

SUB = lib.merged_table()
SUBS = sorted(SUB["cluster"].unique())
NCOL = {s: SUBTYPE_COLORS.get(s, PAL["grey"]) for s in SUBS}
S2C = dict(zip(SUB["sample"], SUB["cluster"]))
MR = pd.read_csv(os.path.join(lib.PRC, "mrna.csv"), index_col=0)


def km_plot(ax, t, e, g, colors, title="", tmax=None):
    t = np.asarray(t, float); e = np.asarray(e, float); g = np.asarray(g)
    keep = ~(np.isnan(t) | np.isnan(e)) & (t > 0)
    t, e, g = t[keep], e[keep], g[keep]
    km = lib.km_curve(t, e, g)
    for name, (xs, ys, n) in km.items():
        ax.step(xs, ys, where="post", lw=1.2, color=colors.get(name, PAL["grey"]),
                label=f"{name} (n={n})")
    chi2, p = lib.logrank(t, e, g)
    ax.set_xlabel("Time (months)", fontsize=7.5)
    ax.set_ylabel("Overall survival", fontsize=7.5)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, tmax or np.nanmax(t) * 1.02)
    ax.text(0.04, 0.06, f"log-rank p = {p:.1e}" if p < 0.01 else f"log-rank p = {p:.3f}",
            transform=ax.transAxes, fontsize=6.8)
    ax.legend(loc="upper right", fontsize=6.4)
    if title:
        ax.set_title(title, fontsize=8)
    return p


def preprocess_survival():
    """Cohort used for the prognostic model.

    The consensus subtypes are defined on the five-omics subset (which is
    necessarily smaller); the prognostic model itself does not depend on the
    subtype labels, so it is trained and evaluated on every TCGA-BRCA sample
    that has both mRNA and overall survival. Subtype labels are carried along
    for the descriptive panels.
    """
    cl = lib.load_clinical()
    samples = [s for s in MR.columns if s[:12] in set(cl.index)]
    d = cl.reindex([s[:12] for s in samples]).reset_index(drop=True)
    d.insert(0, "sample", samples)
    sub = lib.load_subtypes()
    m = dict(zip(sub["sample"], sub["cluster"]))
    d["cluster"] = [m.get(s, "unassigned") for s in samples]
    d = d[d["os_time"].notna() & (d["os_time"] > 0) & d["os_event"].notna()]
    d["os_event"] = d["os_event"].astype(int)
    d = d.drop_duplicates("sample")
    print(f"  model cohort: {len(d)} samples with mRNA + OS "
          f"({int(d['cluster'].ne('unassigned').sum())} in the five-omics set)")
    return d


def main():
    d = preprocess_survival()
    print(f"  survival cohort: {d.shape}")
    Xs = [s for s in d["sample"] if s in MR.columns]
    d = d.set_index("sample").loc[Xs].reset_index()

    # ---- candidate genes: DEG top30 -> univariate Cox
    dep = pd.read_csv(os.path.join(lib.RES, "fig2", "DEG_contrast.csv"))
    cand = dep.reindex(dep["p"].sort_values().index)["gene"].head(300).tolist()
    cand = [g for g in cand if g in MR.index]
    mat = MR.loc[cand, d["sample"]].T
    ucox = lib.cox_univariate(
        pd.concat([mat, d[["os_time", "os_event"]].set_index(d["sample"].values)],
                  axis=1), "os_time", "os_event", cand)
    ucox = ucox.sort_values("p")
    ucox.to_csv(os.path.join(RESD, "univariate_cox.csv"), index=False)
    genes = ucox[ucox["p"] < 0.05]["variable"].tolist()[:30]
    if len(genes) < 5:
        genes = ucox["variable"].head(10).tolist()
    print(f"  {len(genes)} candidate prognostic genes")
    pd.DataFrame({"gene": genes}).to_csv(os.path.join(RESD, "model_genes.csv"),
                                         index=False)

    X = MR.loc[genes, d["sample"]].T.values
    time = d["os_time"].values
    event = d["os_event"].values

    # ---- 101 models
    cache = os.path.join(RESD, "model_scores_101.csv")
    if os.path.exists(cache):
        scores = pd.read_csv(cache)
    else:
        scores = M.eval_models(X, time, event, n_splits=10, n_repeats=1,
                               seed=2026,
                               cache_path=os.path.join(RESD, "model_folds_raw.csv"))
        scores.to_csv(cache, index=False)
    print(scores.head(5).to_string())

    best_name = scores.iloc[0]["model"]
    best = next(m for m in M.build_models() if m[0] == best_name)
    risk = M.fit_final(X, time, event, best)
    d["risk"] = risk
    cut = np.median(risk)
    d["risk_group"] = np.where(risk > cut, "High risk", "Low risk")
    d.to_csv(os.path.join(RESD, "risk_table.csv"), index=False)

    # multivariable Cox on the model genes
    md = pd.DataFrame(X, columns=genes)
    md["os_time"] = time; md["os_event"] = event
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(md, "os_time", "os_event")
    sm = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    sm.columns = ["HR", "lower", "upper", "p"]
    sm.sort_values("HR").to_csv(os.path.join(RESD, "model_forest.csv"))

    # ---- external validation in METABRIC (independent cohort, other platform)
    ext = None
    try:
        import metabric
        Xex = metabric.ensure_expression(genes)
        need = max(5, int(0.5 * len(genes)))
        if Xex.shape[0] >= need:
            # keep only signature genes present on both platforms so that the
            # training and validation matrices share the same feature space
            gx = [g for g in genes if g in Xex.index]
            Xex = Xex.loc[gx]
            genes_ext = gx
            sv = metabric.survival()
            cols = [c for c in Xex.columns if c[:12] in set(sv.index)]
            Xe = Xex[cols]

            def z(df):
                """Per-gene standardisation within a cohort (cross-platform)."""
                zz = df.T.sub(df.mean(axis=1), axis=1).div(
                    df.std(axis=1).replace(0, np.nan), axis=1)
                return zz.T.fillna(0)

            Xtr = z(MR.loc[genes_ext, d["sample"]]).T.values
            Xte = z(Xe).T.values
            et = sv.reindex([c[:12] for c in cols])
            keep = (et["os_time"].notna() & et["os_event"].notna()).values
            if keep.sum() >= 30:
                r_ext = M.fit_predict(Xtr, time, event, Xte[keep], best)
                ext = pd.DataFrame({"patient": np.array([c[:12] for c in cols])[keep],
                                    "os_time": et["os_time"].values[keep],
                                    "os_event": et["os_event"].values[keep].astype(int),
                                    "risk": np.asarray(r_ext, float)})
                ext["risk_group"] = np.where(
                    ext["risk"] > np.median(ext["risk"]), "High risk", "Low risk")
                ext.to_csv(os.path.join(RESD, "external_metabric_risk.csv"),
                           index=False)
                print(f"  METABRIC external validation: n={len(ext)}, "
                      f"genes={Xex.shape[0]}")
    except Exception as e:  # noqa: BLE001
        print("  METABRIC external validation skipped:", e)

    # ================================================== FIG 4
    fig = plt.figure(figsize=(9.2, 7.6))
    gs = GridSpec(3, 3, figure=fig, hspace=0.85, wspace=0.55)

    ax = fig.add_subplot(gs[0, :2])
    s2 = scores.dropna(subset=["mean_C"]).copy()
    s2 = s2.sort_values("mean_C", ascending=False).reset_index(drop=True)
    ax.bar(np.arange(len(s2)), s2["mean_C"], color=PAL["slate"], width=0.85,
           edgecolor="none")
    top = s2.head(3)
    for i in top.index:
        ax.bar(i, s2.loc[i, "mean_C"], color=PAL["red"], width=0.85)
    r = s2.iloc[0]
    ax.text(0, r["mean_C"] + 0.01, r["model"], rotation=90, fontsize=5.8,
            ha="center", va="bottom", color=PAL["red"])
    ax.set_xlabel("Model (10 algorithms, 101 combinations)", fontsize=7.5)
    ax.set_ylabel("Mean CV C-index", fontsize=7.5)
    ax.set_title("Performance of 101 machine-learning combinations", fontsize=8)
    ax.set_ylim(0, 0.88)          # headroom so the rotated top-3 labels fit
    ax.axhline(0.5, color="#999999", lw=0.6, ls="--")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 2])
    f = sm.sort_values("HR")
    for i, (g, r) in enumerate(f.iterrows()):
        c = PAL["red"] if r["HR"] > 1 else PAL["navy"]
        ax.plot([r["lower"], r["upper"]], [i, i], color=c, lw=0.9)
        ax.plot(r["HR"], i, "o", color=c, ms=2.6)
    ax.axvline(1, color="#999999", lw=0.6, ls="--")
    ax.set_yticks(range(len(f)))
    ax.set_yticklabels(f.index, fontsize=5.6)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio", fontsize=7.5)
    ax.set_title("Model coefficients", fontsize=8)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    for g in SUBS:
        v = d.loc[d["cluster"] == g, "risk"]
        ax.hist(v, bins=24, color=NCOL[g], alpha=0.6, lw=0)
    rv = [d.loc[d["cluster"] == g, "risk"].dropna().values for g in SUBS]
    ok = [x for x in rv if len(x) >= 3]
    pv = (float(stats.kruskal(*ok).pvalue) if len(ok) > 2
          else lib.mannwhitney(ok[0], ok[1]) if len(ok) == 2 else 1.0)
    ax.set_xlabel("Risk score", fontsize=7.5); ax.set_ylabel("Samples", fontsize=7.5)
    ax.set_title(f"Risk by subtype ({pval_text(pv)})", fontsize=8)
    ax.legend(SUBS, fontsize=6)
    panel_label(ax, "c")

    # risk-ordered triple
    ax = fig.add_subplot(gs[1, 1:])
    dd = d.sort_values("risk").reset_index(drop=True)
    ax.plot(dd["risk"].values, color=PAL["red"], lw=1.0)
    ax.set_ylabel("Risk score", fontsize=7.5, color=PAL["red"])
    ax.set_xlim(0, len(dd)); ax.set_xticks([])
    ax2t = ax.twinx()
    ax2t.set_yticks([]); ax2t.spines["right"].set_visible(True)
    ax2t.spines["top"].set_visible(False)
    step = max(1, len(dd) // 120)
    surv = 1 - dd["os_event"].values
    ax2t.scatter(np.arange(len(dd))[surv == 0], surv[surv == 0] + 1.15,
                 c=PAL["dark"], s=7, lw=0, marker="s", label="Death")
    ax2t.scatter(np.arange(len(dd))[surv == 1], surv[surv == 1] + 1.15,
                 c=PAL["salmon"], s=7, lw=0, marker="s", label="Alive")
    ax2t.axhline(1.15, color="#BBBBBB", lw=0.5)
    ax2t.set_ylim(-0.4, 2.6)
    ax2t.set_yticks([])
    ax2t.set_ylabel("Survival status", fontsize=7.5)
    ax2t.legend(loc="upper left", fontsize=6, frameon=False, ncol=2)
    ax.set_xlabel("Patients ordered by risk score", fontsize=7.5)
    ax.set_title("Risk-score distribution and survival status", fontsize=8)
    ax.axvline(len(dd) / 2, color="#999999", lw=0.7, ls="--")
    panel_label(ax, "d", x=-0.06)

    ax = fig.add_subplot(gs[2, 0])
    p4 = km_plot(ax, d["os_time"], d["os_event"], d["risk_group"],
                 {"High risk": PAL["red"], "Low risk": PAL["navy"]},
                 "TCGA-BRCA risk stratification")
    panel_label(ax, "e")

    ax = fig.add_subplot(gs[2, 1])
    from lifelines import utils as lifelines_utils
    try:
        from sksurv.metrics import cumulative_dynamic_auc
        y = np.array([(bool(e), tt) for e, tt in zip(d["os_event"], d["os_time"])],
                     dtype=[("event", bool), ("time", float)])
        times = [12, 36, 60]
        auc = cumulative_dynamic_auc(y, y, d["risk"].values, times)[0]
        ax.bar(range(len(times)), auc, color=PAL["teal"], width=0.6)
        for i, a in enumerate(auc):
            ax.text(i, a + 0.01, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([f"{t/12:.0f}-yr" for t in times], fontsize=7)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("Cumulative/dynamic AUC", fontsize=7.5)
        ax.set_title("Time-dependent AUC", fontsize=8)
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"AUC unavailable\n{e}", ha="center", fontsize=6)
        ax.axis("off")
    panel_label(ax, "f")

    # multivariable forest (clinical + risk)
    ax = fig.add_subplot(gs[2, 2])
    md2 = pd.DataFrame({
        "risk_group": (d["risk_group"] == "High risk").astype(int).values,
        "age": pd.to_numeric(d["age"], errors="coerce").values,
        "stage_num": d["stage"].str.extract(r"STAGE ([IV]+)")[0].map(
            {"I": 1, "II": 2, "III": 3, "IV": 4}).values,
        "os_time": time, "os_event": event}).dropna()
    try:
        cph2 = CoxPHFitter(penalizer=0.05)
        cph2.fit(md2, "os_time", "os_event")
        s3 = cph2.summary[["exp(coef)", "exp(coef) lower 95%",
                           "exp(coef) upper 95%", "p"]]
        s3.columns = ["HR", "lower", "upper", "p"]
        for i, (g, r) in enumerate(s3.iloc[::-1].iterrows()):
            c = PAL["red"] if r["HR"] > 1 else PAL["navy"]
            ax.plot([r["lower"], r["upper"]], [i, i], color=c, lw=0.9)
            ax.plot(r["HR"], i, "o", color=c, ms=2.8)
        ax.set_yticks(range(len(s3)))
        ax.set_yticklabels(s3.index[::-1], fontsize=6)
        ax.axvline(1, color="#999999", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Hazard ratio (multivariable)", fontsize=7.5)
        ax.set_title("Independence of the risk score", fontsize=8)
        s3.to_csv(os.path.join(RESD, "multivariable_cox.csv"))
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"Cox failed\n{e}", ha="center", fontsize=6)
        ax.axis("off")
    panel_label(ax, "g")
    save_fig(fig, FIGD, "Fig4")
    print("Fig4 done")

    # ================================================== FIG 5
    # benchmark
    bench = {}
    bench["This model"] = d["risk"].values
    for g in SUBS:
        bench[f"Subtype {g}"] = (d["cluster"] == g).astype(float).values
    bench["Age"] = pd.to_numeric(d["age"], errors="coerce").values
    bench["Stage"] = d["stage"].str.extract(r"STAGE ([IV]+)")[0].map(
        {"I": 1, "II": 2, "III": 3, "IV": 4}).values
    bench["Grade"] = pd.to_numeric(d["grade"], errors="coerce").values
    pam = {"LumA": 1, "LumB": 2, "Her2": 3, "Basal": 4, "Normal": 5}
    bench["PAM50"] = d["pam50"].map(pam).values
    prolif = [g for g in ["MKI67", "AURKA", "BUB1", "CCNB1", "CDK1", "TOP2A",
                          "RRM2", "UBE2C", "BIRC5", "CCNA2"] if g in MR.index]
    bench["Proliferation"] = MR.loc[prolif, d["sample"]].mean(axis=0).values
    imm = [g for g in ["PTPRC", "CD3D", "CD8A", "CD2", "CXCL9", "GZMA"] if g in MR.index]
    bench["Immune"] = MR.loc[imm, d["sample"]].mean(axis=0).values
    rows = []
    for k, v in bench.items():
        c = M.cindex(time, event, np.asarray(v, float))
        rows.append({"predictor": k, "C_index": c})
    bd = pd.DataFrame(rows).sort_values("C_index", ascending=False)
    bd.to_csv(os.path.join(RESD, "benchmark.csv"), index=False)

    fig = plt.figure(figsize=(9.2, 7.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.6, wspace=0.65, height_ratios=[1, 1.05])

    ax = fig.add_subplot(gs[0, :2])
    yy = np.arange(len(bd))[::-1]
    cols = [PAL["red"] if p == "This model" else PAL["slate"] for p in bd["predictor"]]
    ax.barh(yy, bd["C_index"], color=cols, height=0.62)
    ax.set_yticks(yy); ax.set_yticklabels(bd["predictor"], fontsize=6.5)
    for y, v in zip(yy, bd["C_index"]):
        if not np.isnan(v):
            ax.text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=6)
    ax.axvline(0.5, color="#999999", lw=0.6, ls="--")
    ax.set_xlabel("C-index", fontsize=7.5)
    ax.set_title("Benchmark against clinical and molecular predictors", fontsize=8)
    panel_label(ax, "a")

    # nomogram - proper layout: per-variable scales with tick values, a
    # total-points axis and a mapped 3-year overall-survival axis
    ax = fig.add_subplot(gs[0, 2])
    try:
        nom = md2.copy()
        cph3 = CoxPHFitter(penalizer=0.05)
        cph3.fit(nom, "os_time", "os_event")
        coefs = cph3.params_
        assert (coefs > 0).all(), "nomogram assumes positive coefficients"
        rngv = {c: (nom[c].min(), nom[c].max()) for c in coefs.index}
        pts = {c: coefs[c] * (rngv[c][1] - rngv[c][0]) for c in coefs.index}
        mx = max(pts.values())
        tpt_max = sum(pts.values()) / mx * 100          # full-scale total points
        vlab = {"risk_group": "Risk group", "age": "Age (years)",
                "stage_num": "Stage (I-IV)"}
        ypos = np.arange(len(pts))[::-1]
        for y, c in zip(ypos, pts):
            v0, v1 = rngv[c]
            ticks = np.array([v0, v1]) if c == "risk_group" else \
                np.linspace(v0, v1, 5)
            tpts = coefs[c] * (ticks - v0) / mx * 100
            ax.plot(tpts, [y] * len(ticks), color=PAL["dark"], lw=0.8)
            ax.plot(tpts, [y] * len(ticks), "|", color=PAL["dark"], ms=4)
            for tp, tv in zip(tpts, ticks):
                ax.text(tp, y + 0.22, f"{tv:g}", ha="center",
                        va="bottom", fontsize=4.8, color="#333333")
            ax.text(-5, y, vlab.get(c, c), ha="right", va="center", fontsize=6)
        ax.set_xlim(-34, 104); ax.set_ylim(-1.0, len(pts) - 0.1)
        ax.set_yticks([]); ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Points", fontsize=7.5)
        # --- bottom axes: total points -> 3-year OS via the baseline hazard
        lp_c = cph3.predict_log_partial_hazard(nom)     # centred LP
        tmin, tmax = lp_c.min(), lp_c.max()
        bs = cph3.baseline_survival_
        s36 = float(np.interp(36, bs.index.values, bs.iloc[:, 0].values))
        tp_ticks = np.array([0, 25, 50, 75, 100]) * tpt_max / 100
        ax.set_xticks(tp_ticks)
        ax.set_xlabel("Total points", fontsize=7.5)
        ax3 = ax.twiny()
        ax3.set_xlim(-34, 104)
        lp_ticks = tmin + np.array([0, 25, 50, 75, 100]) / 100 * (tmax - tmin)
        s_ticks = s36 ** np.exp(lp_ticks)
        ax3.set_xticks([0, 25, 50, 75, 100])
        ax3.set_xticklabels([f"{s:.2f}" for s in s_ticks], fontsize=6)
        ax3.set_xlabel("3-year overall survival", fontsize=7)
        ax3.spines["top"].set_visible(False)
        ax3.xaxis.set_ticks_position("bottom")
        ax3.xaxis.set_label_position("bottom")
        ax3.spines["bottom"].set_position(("outward", 20))
        ax3.tick_params(axis="x", colors=PAL["red"])
        ax3.set_xlabel("3-year overall survival", fontsize=7, color=PAL["red"])
        ax.set_title("Nomogram", fontsize=8, pad=10)
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"nomogram failed\n{e}", ha="center", fontsize=6)
        ax.axis("off")
    panel_label(ax, "b", x=-0.28)

    # calibration - deciles, axes zoomed to the observed probability range
    ax = fig.add_subplot(gs[1, 0])
    try:
        from lifelines import CoxPHFitter as CPH
        dd2 = d.copy()
        dd2["lp"] = dd2["risk"]
        q = pd.qcut(dd2["risk"], 10, labels=False, duplicates="drop")
        pred, obs = [], []
        # unpenalised refit - a penalised fit shrinks the linear predictor and
        # artificially compresses the predicted-probability range
        base = CPH(penalizer=0.0)
        base.fit(dd2[["risk", "os_time", "os_event"]], "os_time", "os_event")
        for k in sorted(pd.unique(q)):
            sub = dd2[q == k]
            t0 = 36
            s0 = base.predict_survival_function(sub, times=[t0]).values.mean()
            pred.append(1 - s0)
            # observed KM at t0
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter().fit(sub["os_time"], sub["os_event"])
            obs.append(1 - float(kmf.predict(t0)))
        pred, obs = np.array(pred), np.array(obs)
        lim = max(pred.max(), obs.max()) * 1.25
        ax.plot([0, lim], [0, lim], color="#999999", lw=0.7, ls="--")
        ax.plot(pred, obs, "o-", color=PAL["red"], ms=3.5, lw=1.0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Predicted 3-year mortality", fontsize=7.5)
        ax.set_ylabel("Observed 3-year mortality", fontsize=7.5)
        ax.set_title("Calibration (deciles)", fontsize=8)
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"calibration failed\n{e}", ha="center", fontsize=6)
        ax.axis("off")
    panel_label(ax, "c")

    # DCA - thresholds must act on the model's predicted 3-year mortality,
    # not on percentile ranks of the risk score
    ax = fig.add_subplot(gs[1, 1])
    try:
        t0 = 36
        dd2 = d.copy()
        # predicted 3-year mortality per patient from the multivariable model
        # (cph3/nom fitted in panel b share the index with md2 -> d rows)
        surv36 = cph3.predict_survival_function(
            nom[list(coefs.index)], times=[t0]).iloc[0]
        dd2 = dd2.loc[nom.index].copy()
        dd2["p_mort"] = 1 - surv36.values
        n_all = len(dd2)
        ev_all = ((dd2["os_time"] <= t0) & (dd2["os_event"] == 1)).sum()

        def net_benefit(prob, threshold):
            hi = prob >= threshold
            if hi.sum() == 0:
                return 0.0
            flag = dd2[hi]
            ev = ((flag["os_time"] <= t0) & (flag["os_event"] == 1)).sum()
            nb = ev / n_all - (hi.sum() - ev) / n_all * (threshold / (1 - threshold))
            return nb

        pt = np.linspace(0.005, min(0.6, dd2["p_mort"].max() * 1.1), 80)
        nb_model = [net_benefit(dd2["p_mort"].values, t) for t in pt]
        nb_all = [ev_all / n_all - (n_all - ev_all) / n_all * (t / (1 - t))
                  for t in pt]
        nb_none = np.zeros_like(pt)
        ax.plot(pt, nb_model, color=PAL["red"], lw=1.2, label="This model")
        ax.plot(pt, nb_all, color=PAL["navy"], lw=1.0, ls="--", label="Treat all")
        ax.plot(pt, nb_none, color="#999999", lw=1.0, ls=":", label="Treat none")
        ax.set_ylim(-0.02, max(0.02, np.nanmax(nb_model)) * 1.25)
        ax.set_xlim(0, pt.max())
        ax.set_xlabel("Threshold probability", fontsize=7.5)
        ax.set_ylabel("Net benefit (3-year)", fontsize=7.5)
        ax.set_title("Decision-curve analysis", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper right")
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"DCA failed\n{e}", ha="center", fontsize=6)
        ax.axis("off")
    panel_label(ax, "d")

    # external validation (METABRIC) - falls back to the PAM50 panel
    ax = fig.add_subplot(gs[1, 2])
    if ext is not None and len(ext) > 20:
        km_plot(ax, ext["os_time"], ext["os_event"], ext["risk_group"],
                {"High risk": PAL["red"], "Low risk": PAL["navy"]},
                "METABRIC (independent cohort)")
        ax.text(0.04, 0.92, "external validation", transform=ax.transAxes,
                fontsize=6, color="#666666")
    else:
        grps = []
        for g in SUBS:
            for p in ["LumA", "LumB", "Her2", "Basal", "Normal"]:
                v = d.loc[(d["cluster"] == g) & (d["pam50"] == p), "risk"]
                if len(v) >= 3:
                    grps.append((f"{g}\n{p}", v.values, NCOL[g]))
        pos = np.arange(len(grps))
        for i, (nm, v, c) in enumerate(grps):
            ax.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v,
                       s=2.2, color=c, alpha=0.7, lw=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#1B1B1B", lw=1.0)
        ax.set_xticks(pos)
        ax.set_xticklabels([g[0] for g in grps], fontsize=4.6, rotation=90)
        ax.set_ylabel("Risk score", fontsize=7.5)
        ax.set_title("Risk score within PAM50 classes", fontsize=8)
    panel_label(ax, "e", x=-0.22)
    save_fig(fig, FIGD, "Fig5")
    print("Fig5 done")

    with open(os.path.join(RESD, "model_pickle.pkl"), "wb") as f:
        pickle.dump({"genes": genes, "best_model": best_name, "risk": risk}, f)
    print("DONE")


if __name__ == "__main__":
    main()
