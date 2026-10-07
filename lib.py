"""
lib.py - shared analysis helpers: data loading, survival statistics,
differential expression, enrichment, immune infiltration.
"""
import itertools
import json
import os
import tempfile
import warnings

# ASCII temp dirs (see 10_cluster.py) -- non-ASCII user names break joblib's
# memmapping folder and multiprocessing's resource tracker on Windows.
_TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tmp")
os.makedirs(_TMP, exist_ok=True)
for _v in ("TMPDIR", "TEMP", "TMP", "JOBLIB_TEMP_FOLDER"):
    os.environ[_v] = _TMP
tempfile.tempdir = _TMP

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRC = os.path.join(PROJ, "data", "processed")
RAW = os.path.join(PROJ, "data", "raw")
RES = os.path.join(PROJ, "results")
FIG = os.path.join(PROJ, "figures")


# ------------------------------------------------------------------- clinical
def load_subtypes(tag=""):
    p = os.path.join(RES, "cluster", f"subtypes{tag}.csv")
    d = pd.read_csv(p)
    d["patient"] = d["sample"].str[:12]
    return d


def subtype_list(tag=""):
    return sorted(load_subtypes(tag)["cluster"].unique())


def contrast_pair(tag="", cache=True):
    """The two transcriptomically most divergent subtypes.

    Pairwise panels of the manuscript (DEG, GSEA, CPTAC and single-cell
    contrasts) need a single, reproducible contrast axis instead of a
    hard-coded BC1-vs-BC2 assumption, so the pair is derived from the data:
    subtype centroids are computed over the 2,000 most variable protein-coding
    genes and the pair with the lowest Pearson correlation is returned.
    """
    out = os.path.join(RES, "cluster", f"contrast_pair{tag}.json")
    if cache and os.path.exists(out):
        d = json.load(open(out))
        return tuple(d["pair"])
    sub = load_subtypes(tag)
    mrna = pd.read_csv(os.path.join(PRC, "mrna.csv"), index_col=0)
    labs = sorted(sub["cluster"].unique())
    if len(labs) < 2:
        return (labs[0], labs[0]) if labs else ("BC1", "BC2")
    top = mrna.var(axis=1).sort_values(ascending=False).head(2000).index
    cen = {}
    for g in labs:
        cols = [s for s in sub.loc[sub["cluster"] == g, "sample"]
                if s in mrna.columns]
        cen[g] = mrna.loc[top, cols].mean(axis=1)
    R = pd.DataFrame(cen).corr()
    best, bv = (labs[0], labs[1]), 2.0
    for a, b in itertools.combinations(labs, 2):
        if R.loc[a, b] < bv:
            best, bv = (a, b), R.loc[a, b]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"pair": list(best), "pearson_r": float(bv),
               "centroid_correlation": R.round(4).to_dict()},
              open(out, "w"), indent=1)
    print(f"  contrast pair -> {best[0]} vs {best[1]} (r = {bv:.3f})")
    return best


def load_clinical():
    """Patient-level clinical table with harmonised survival columns."""
    tcga = pd.read_csv(os.path.join(PRC, "tcga_clinical_patient.csv"), index_col=0)
    tcga.index.name = "patient"
    sv = pd.read_csv(os.path.join(PRC, "tcga_survival_xena.csv"))
    sv["patient"] = sv["_PATIENT"]
    sv = sv.drop_duplicates("patient").set_index("patient")

    df = tcga.copy()
    df["OS_MONTHS"] = pd.to_numeric(df.get("OS_MONTHS"), errors="coerce")
    df["OS_STATUS"] = df.get("OS_STATUS").astype(str)
    df["os_event"] = df["OS_STATUS"].str.startswith("1").astype(int)
    df["os_time"] = df["OS_MONTHS"]
    # fall back on the harmonised Xena survival file where missing
    miss = df["os_time"].isna() | (df["os_time"] <= 0)
    # the Xena survival file stores time in DAYS while OS_MONTHS is in months
    df.loc[miss, "os_time"] = (sv.reindex(df.index[miss])["OS.time"].values
                               / 30.4375)
    df.loc[miss, "os_event"] = sv.reindex(df.index[miss])["OS"].values
    # ---- harmonised survival endpoints (months, 1 = event) -----------------
    endpoints = [("OS", "OS_MONTHS", "OS_STATUS"),
                 ("PFS", "PFS_MONTHS", "PFS_STATUS"),
                 ("DSS", "DSS_MONTHS", "DSS_STATUS"),
                 ("DFS", "DFS_MONTHS", "DFS_STATUS")]
    for tag, tcol, scol in endpoints:
        if tcol not in df.columns:
            continue
        tt = pd.to_numeric(df[tcol], errors="coerce")
        ss = df[scol].astype(str) if scol in df.columns else pd.Series("", index=df.index)
        ev = np.where(ss.str.match(r"^1"), 1.0,
                      np.where(ss.str.match(r"^0"), 0.0, np.nan))
        df[f"{tag.lower()}_time"] = tt
        df[f"{tag.lower()}_event"] = ev
    # OS is the reference endpoint
    df["os_time"] = df.get("os_time", df["os_time"] if "os_time" in df else np.nan)

    df["age"] = pd.to_numeric(df.get("AGE"), errors="coerce")
    df["stage"] = df.get("AJCC_PATHOLOGIC_TUMOR_STAGE").astype(str)
    df["pam50"] = df.get("SUBTYPE").astype(str).str.replace("BRCA_", "", regex=False)
    df["grade"] = df.get("GRADE").astype(str) if "GRADE" in df else np.nan
    return df


def load_sample_clinical():
    s = pd.read_csv(os.path.join(PRC, "tcga_clinical_sample.csv"), index_col=0)
    s.index.name = "sample"
    return s


def merged_table(tag=""):
    sub = load_subtypes(tag)
    cl = load_clinical()
    m = sub.merge(cl, left_on="patient", right_index=True, how="left")
    return m


# ------------------------------------------------------------------- survival
def km_curve(time, event, group, tmax=None):
    """Kaplan-Meier estimate. Returns step arrays per group."""
    out = {}
    for g in pd.unique(group):
        m = np.asarray(group) == g
        t, e = np.asarray(time)[m], np.asarray(event)[m]
        ok = ~(pd.isna(t) | pd.isna(e))
        t, e = t[ok], e[ok]
        order = np.argsort(t)
        t, e = t[order], e[order]
        n = len(t)
        s = 1.0
        xs, ys = [0.0], [1.0]
        for i in range(n):
            if e[i] == 1:
                s *= (n - i - 1) / (n - i)
                xs.append(t[i])
                ys.append(s)
        xs.append(t[-1] if n else 0)
        ys.append(ys[-1])
        out[g] = (np.array(xs), np.array(ys), n)
    return out


def logrank(time, event, group):
    """Log-rank test (2+ groups)."""
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    group = np.asarray(group)
    ok = ~(np.isnan(time) | np.isnan(event))
    time, event, group = time[ok], event[ok], group[ok]
    gs = pd.unique(group)
    if len(gs) < 2:
        return 1.0, 1.0
    times = np.unique(time[event == 1])
    O1 = E1 = V = 0.0
    for t in times:
        at_risk = time >= t
        n_j = at_risk.sum()
        d_j = ((time == t) & (event == 1)).sum()
        for g in gs[:-1]:
            n1 = (at_risk & (group == g)).sum()
            d1 = ((time == t) & (event == 1) & (group == g)).sum()
            O1 += d1
            E1 += d_j * n1 / n_j if n_j else 0
            if n_j > 1:
                V += (d_j * (n1 / n_j) * (1 - n1 / n_j) *
                      (n_j - d_j) / (n_j - 1))
    if V <= 0:
        return 1.0, 0.0
    chi2 = (O1 - E1) ** 2 / V
    return float(chi2), float(stats.chi2.sf(chi2, len(gs) - 1))


def cox_univariate(df, time_col, event_col, covars):
    """Univariate Cox via lifelines; returns HR table."""
    from lifelines import CoxPHFitter
    rows = []
    for c in covars:
        try:
            d = df[[c, time_col, event_col]].dropna()
            d = d[d[time_col] > 0]
            if d[c].nunique() < 2 or len(d) < 20:
                continue
            cph = CoxPHFitter()
            cph.fit(d, duration_col=time_col, event_col=event_col)
            s = cph.summary.loc[c]
            rows.append({"variable": c, "HR": s["exp(coef)"],
                         "lower": s["exp(coef) lower 95%"],
                         "upper": s["exp(coef) upper 95%"], "p": s["p"],
                         "n": len(d)})
        except Exception as e:  # noqa: BLE001
            print(f"   cox failed {c}: {e}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ DEG / stats
def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        v = p[o[i]] * n / (i + 1)
        prev = min(prev, v)
        q[o[i]] = prev
    return q


def diff_expr(mat, group, g1, g2, min_frac=0.1):
    """Welch t-test + BH FDR between two sample groups (columns of mat)."""
    a = mat.loc[:, [c for c in mat.columns if group.get(c) == g1]]
    b = mat.loc[:, [c for c in mat.columns if group.get(c) == g2]]
    if a.shape[1] < 3 or b.shape[1] < 3:
        return pd.DataFrame()
    mask = (a > 0).mean(axis=1) >= min_frac
    mask |= (b > 0).mean(axis=1) >= min_frac
    a, b = a.loc[mask], b.loc[mask]
    t, p = stats.ttest_ind(a.values, b.values, axis=1, equal_var=False)
    fc = a.mean(axis=1).values - b.mean(axis=1).values
    q = bh(np.nan_to_num(p, nan=1.0))
    d = pd.DataFrame({"gene": a.index, "log2FC": fc, "p": p, "FDR": q,
                      "mean_g1": a.mean(axis=1).values,
                      "mean_g2": b.mean(axis=1).values})
    return d.sort_values("p")


def mannwhitney(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return 1.0
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


# ------------------------------------------------------------------ enrichment
def load_gene_sets(collection="h.all"):
    try:
        import gseapy
        gs = gseapy.get_library_name()
        lib = gseapy.read_gmt
        if collection == "h.all":
            d = gseapy.get_library("MSigDB_Hallmark_2020")
        elif collection == "kegg":
            d = gseapy.get_library("KEGG_2021_Human")
        elif collection == "reactome":
            d = gseapy.get_library("Reactome_2022")
        else:
            d = gseapy.get_library(collection)
        return d
    except Exception as e:  # noqa: BLE001
        print("  gene set load failed:", e)
        return {}


def ssgsea(mat, gene_sets, min_size=3, max_size=500, sample_norm=True):
    """Vectorised single-sample GSEA (Barbie 2009) on a gene x sample matrix."""
    import gseapy
    res = gseapy.ssgsea(data=mat, gene_sets=gene_sets, outdir=None,
                        min_size=min_size, max_size=max_size,
                        sample_norm_method="rank" if sample_norm else "none",
                        no_plot=True, threads=4)
    d = res.res2d
    scores = res.res2d.pivot_table(index="Term", columns="Name",
                                   values="NES", aggfunc="first")
    return scores


def gsea_prerank(dep, gene_sets, min_size=10, max_size=500):
    """GSEA on a ranked DE table (gene, log2FC)."""
    import gseapy
    r = dep[["gene", "log2FC"]].dropna().sort_values("log2FC", ascending=False)
    pre = r.set_index("gene")["log2FC"]
    try:
        res = gseapy.prerank(rnk=pre, gene_sets=gene_sets, min_size=min_size,
                             max_size=max_size, permutation_num=300,
                             outdir=None, seed=20260911, no_plot=True, threads=4)
        return res.res2d
    except Exception as e:  # noqa: BLE001
        print("  prerank failed:", e)
        return pd.DataFrame()


def hallmark_pick(hall, spec):
    """Resolve Hallmark set names across naming conventions.

    Enrichr's MSigDB_Hallmark_2020 library uses plain names such as
    'E2F Targets', while other sources use 'HALLMARK_E2F_TARGETS'. `spec`
    maps a display label to substrings that must all occur in the set name
    (case-insensitive, ignoring separators); the first match wins.
    """
    out = {}
    for label, parts in spec.items():
        parts = [p.lower() for p in parts]
        for k in hall:
            kl = k.lower().replace("_", " ").replace("-", " ")
            if all(p in kl for p in parts):
                out[label] = list(hall[k])
                break
    return out


# --------------------------------------------------- nearest template prediction
def ntp_templates(mrna, groups, n_genes=200):
    """Per-subtype nearest-template-prediction templates (Efron & Tibshirani 2007).

    For every subtype the genes with the largest positive t-statistic against
    all other subtypes are selected; their t-statistics, scaled to [-1, 1],
    form the template vector.  Returns (templates, gene_lists).
    """
    if len(groups) != mrna.shape[1]:
        raise ValueError(
            f"ntp_templates: {len(groups)} group labels for "
            f"{mrna.shape[1]} columns -- align the matrix to the cohort first")
    templates, gene_lists = {}, {}
    for g in pd.unique(groups):
        m = np.asarray(groups) == g
        a, b = mrna.values[:, m], mrna.values[:, ~m]
        t = stats.ttest_ind(a, b, axis=1, equal_var=False)[0]
        s = pd.Series(np.nan_to_num(t), index=mrna.index).sort_values(
            ascending=False).head(n_genes)
        if s.iloc[0] <= 0:
            continue
        scale = s.abs().max()
        templates[g] = s / (scale if scale > 0 else 1.0)
        gene_lists[g] = s.index.tolist()
    return templates, gene_lists


def ntp_predict(templates, X):
    """Score every column of X (genes x samples) against each template.

    Genes are z-scored within each sample and correlated with the template
    vector, giving one correlation per sample and subtype.
    """
    Z = X.sub(X.mean(axis=1), axis=0).div(
        X.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    C = np.full((X.shape[1], len(templates)), np.nan)
    for j, (g, T) in enumerate(templates.items()):
        idx = [i for i, gg in enumerate(X.index) if gg in T.index]
        if len(idx) < 10:
            continue
        zv = Z.values[idx]
        tv = T.reindex(X.index[idx]).values.astype(float)
        tv = np.nan_to_num(tv)
        tv = tv - tv.mean()
        num = zv.T @ tv
        den = np.linalg.norm(zv, axis=0) * np.linalg.norm(tv)
        C[:, j] = np.where(den > 0, num / den, 0.0)
    return C


# ------------------------------------------------------------------- markers
CHAROENTONG = {
    "aDC": ["CD1C", "CLEC9A", "CLEC10A", "ITGAX"],
    "B_cells": ["CD19", "MS4A1", "CD79A", "CD79B", "TNFRSF17"],
    "CD8_T_cells": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "IFNG"],
    "Cytotoxic_cells": ["GNLY", "NKG7", "KLRD1", "KLRK1"],
    "DC": ["CD1C", "HLA-DRA", "HLA-DPA1", "ITGAX", "CD1E"],
    "Eosinophils": ["PRG2", "CLC", "IL5RA", "CCR3"],
    "iDC": ["ITGAX", "CD1C", "HLA-DRA", "HLA-DPA1"],
    "Macrophages": ["CD68", "CD163", "CSF1R", "MRC1", "MSR1"],
    "Mast_cells": ["TPSAB1", "CPA3", "MS4A2", "TPSB2"],
    "Monocytes": ["S100A8", "S100A9", "CD14", "LYZ", "FCN1"],
    "Neutrophils": ["FCGR3B", "CSF3R", "CXCR2", "S100A8"],
    "NK_cells": ["KLRF1", "KLRD1", "NCAM1", "NKG7", "GNLY"],
    "NKT": ["NKG7", "KLRD1", "CD3E", "GNLY"],
    "pDC": ["LILRA4", "CLEC4C", "IL3RA", "GZMB"],
    "T_helper_cells": ["CD4", "IL21", "CD40LG", "ICOS"],
    "Tfh": ["CXCR5", "ICOS", "BCL6", "PDCD1"],
    "Th1_cells": ["TBX21", "IFNG", "TNF", "CXCR3"],
    "Th2_cells": ["GATA3", "IL4", "IL5", "IL13", "CCR4"],
    "Th17_cells": ["RORC", "IL17A", "IL17F", "CCR6"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "Tgd": ["TRDC", "TRGC1", "TRGC2", "KLRD1"],
    "CD4_T_cells": ["CD4", "CD40LG", "IL7R", "CCR7"],
    "Memory_B_cells": ["CD27", "MS4A1", "CD79A", "TNFRSF13B"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5"],
    "Fibroblasts": ["COL1A1", "COL1A2", "DCN", "FAP", "PDGFRA"],
    "Pericytes": ["RGS5", "PDGFRB", "ACTA2", "NOTCH3"],
}
SENMAYO = ["CDKN1A", "CDKN2A", "IL6", "CXCL8", "GLB1", "SERPINE1", "TP53BP1",
           "ATM", "CHEK1", "CHEK2", "MDM2", "MAPK14", "SERPINB2", "IGFBP7",
           "VEGFA", "CCNA2", "MKI67", "LMNB1", "SIRT1", "SIRT6", "FOXO3",
           "TGFB1", "TGFBR1", "TGFBR2", "B2M", "H2AX", "TERF2", "RAD51",
           "PCNA", "POLA1", "EEF1E1", "ETS2", "RBL1", "RBL2", "RB1", "E2F1",
           "E2F2", "E2F3", "BUB1B", "PRKDC", "H2AFX", "PCNA", "TP53"]
