"""
04_preprocess.py - Build cleaned multi-omics matrices for TCGA-BRCA.

Inputs (data/raw/):
  TCGA-BRCA.star_counts.tsv.gz   log2(count+1), Ensembl gene id x sample
  TCGA-BRCA.mirna.tsv.gz         log2 rpm, miRNA x sample
  TCGA-BRCA.methylation450.tsv.gz beta values, probe x sample
  TCGA-BRCA.survival.tsv.gz      OS time / event
  gdc_maf/*.maf.gz               masked somatic mutation MAFs
  hgnc_complete_set.txt          Ensembl id -> symbol / locus_type
  cbioportal/*_clinical_full.json  clinical (subtype, stage, survival)

Outputs (data/processed/):
  gene_annotation.tsv, mrna.csv, lncrna.csv, mirna.csv,
  methyl_top*.csv, mut_matrix.csv, mut_maf.parquet, clinical_tcga.csv
"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "data", "raw")
PRC = os.path.join(PROJ, "data", "processed")
os.makedirs(PRC, exist_ok=True)
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def sample_cols(df):
    return [c for c in df.columns if c.startswith("TCGA-")]


def norm_cols(df):
    """Harmonise sample barcodes to the 15-character TCGA sample id
    (TCGA-XX-XXXX-01A -> TCGA-XX-XXXX-01), keep primary tumours only,
    average technical replicates."""
    df = df[sample_cols(df)]
    names = pd.Series(df.columns)
    keep = names.str[13:15].astype(str).str.fullmatch(r"0[1-9]").values
    df = df.loc[:, keep]
    df.columns = [c[:15] for c in df.columns]
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).mean().T
    return df


PROTEIN_CODING = "gene with protein product"
LNCRNA_TYPES = ["RNA, long non-coding", "lincRNA", "antisense",
                "processed_transcript", "lncRNA", "non_coding",
                "3prime_overlapping_ncRNA", "macro_lncRNA", "sense_intronic",
                "sense_overlapping"]


def get_gene_annotation():
    """Ensembl gene id -> symbol + biotype (HGNC primary, GDC supplement)."""
    out = os.path.join(PRC, "gene_annotation.tsv")
    if os.path.exists(out):
        return pd.read_csv(out, sep="\t")
    log("building gene annotation (HGNC + GDC) ...")
    h = pd.read_csv(os.path.join(RAW, "hgnc_complete_set.txt"), sep="\t",
                    low_memory=False,
                    usecols=["symbol", "locus_type", "ensembl_gene_id"])
    h = h.dropna(subset=["ensembl_gene_id"])
    rows = []
    for sym, ltype, eids in zip(h["symbol"], h["locus_type"], h["ensembl_gene_id"]):
        for e in str(eids).split("|"):
            e = e.strip()
            if e.startswith("ENSG"):
                rows.append((e, sym, ltype))
    hg = pd.DataFrame(rows, columns=["gene_id", "symbol", "biotype"]).drop_duplicates("gene_id")
    log(f"  HGNC mapped {len(hg)} Ensembl ids; "
        f"protein_coding={int((hg.biotype == PROTEIN_CODING).sum())}, "
        f"lncRNA={int(hg.biotype.isin(LNCRNA_TYPES).sum())}")
    try:
        url = ("https://api.gdc.cancer.gov/genes?"
               "fields=gene_id,symbol,biotype&size=25000&format=JSON")
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.load(r)
        gd = pd.DataFrame(d["data"]["hits"])[["gene_id", "symbol", "biotype"]]
        gd = gd[~gd["gene_id"].isin(hg["gene_id"])]
        ann = pd.concat([hg, gd], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        log(f"  GDC supplement failed ({e}); using HGNC only")
        ann = hg
    ann = ann.drop_duplicates("gene_id")
    ann.to_csv(out, sep="\t", index=False)
    log(f"  total {len(ann)} genes annotated")
    return ann


def load_expression(ann):
    out_m = os.path.join(PRC, "mrna.csv")
    out_l = os.path.join(PRC, "lncrna.csv")
    if os.path.exists(out_m) and os.path.exists(out_l):
        return pd.read_csv(out_m, index_col=0), pd.read_csv(out_l, index_col=0)
    log("loading star_counts ...")
    ex = pd.read_csv(os.path.join(RAW, "TCGA-BRCA.star_counts.tsv.gz"),
                     sep="\t", index_col=0)
    ex.index = ex.index.str.split(".").str[0]
    ex = ex[~ex.index.duplicated()]
    log(f"  raw matrix {ex.shape}")

    bio = ann.set_index("gene_id")
    sym = bio["symbol"].reindex(ex.index)
    bty = bio["biotype"].reindex(ex.index)
    keep = sym.notna() & bty.notna()
    ex, sym, bty = ex.loc[keep], sym[keep], bty[keep]

    ex = norm_cols(ex)
    ex = ex.assign(_sym=sym.values, _bty=bty.values)
    cols = sample_cols(ex)
    ex["_mean"] = ex[cols].mean(axis=1)
    ex = ex.sort_values("_mean", ascending=False).drop_duplicates("_sym")
    bty_final = ex["_bty"].to_numpy()
    ex = ex.drop(columns=["_mean", "_bty"]).set_index("_sym")
    ex.index.name = "gene"

    mrna = ex.loc[bty_final == PROTEIN_CODING]
    lncrna = ex.loc[np.isin(bty_final, LNCRNA_TYPES)]
    log(f"  mRNA {mrna.shape}   lncRNA {lncrna.shape}")
    mrna.to_csv(out_m)
    lncrna.to_csv(out_l)
    return mrna, lncrna


def load_mirna():
    out = os.path.join(PRC, "mirna.csv")
    if os.path.exists(out):
        return pd.read_csv(out, index_col=0)
    log("loading miRNA ...")
    mi = pd.read_csv(os.path.join(RAW, "TCGA-BRCA.mirna.tsv.gz"), sep="\t", index_col=0)
    mi = norm_cols(mi)
    log(f"  miRNA {mi.shape}")
    mi.to_csv(out)
    return mi


def load_methylation(top_n=10000, chunk=25000):
    """Stream the 450K beta matrix twice: pass 1 collects per-CpG mean/variance
    (chunks concatenated, not summed), pass 2 keeps the selected rows."""
    out = os.path.join(PRC, f"methyl_top{top_n}.csv")
    if os.path.exists(out):
        return pd.read_csv(out, index_col=0)
    path = os.path.join(RAW, "TCGA-BRCA.methylation450.tsv.gz")
    log(f"streaming methylation (selecting top {top_n} variable CpGs) ...")
    s1, s2, cnt, idx = [], [], [], []
    n_samp = None
    for ch in pd.read_csv(path, sep="\t", index_col=0, chunksize=chunk):
        ch = norm_cols(ch)
        cols = sample_cols(ch)
        if n_samp is None:
            n_samp = len(cols)
        v = ch[cols].astype(np.float32).values
        s1.append(np.nansum(v, axis=1, dtype=np.float64))
        s2.append(np.nansum(v.astype(np.float64) ** 2, axis=1))
        cnt.append((~np.isnan(v)).sum(axis=1))
        idx.append(ch.index.to_numpy())
        log(f"   pass1: {sum(len(a) for a in idx)} CpGs scanned")
    s1 = np.concatenate(s1)
    s2 = np.concatenate(s2)
    cnt = np.concatenate(cnt)
    idx = np.concatenate(idx)
    obs = np.maximum(cnt, 1)
    mean = s1 / obs
    var = s2 / obs - mean ** 2
    # coverage is judged against the number of SAMPLES (cols), not the number
    # of CpGs scanned -- using the CpG count silently disabled this filter.
    enough = (cnt / n_samp) >= 0.8
    var = np.where(enough & (mean > 0.05) & (mean < 0.95), var, -1.0)
    log(f"   CpGs with >=80% sample coverage and 0.05<beta<0.95: "
        f"{int((var >= 0).sum())} / {len(var)}")
    top = np.argsort(var)[::-1][:top_n]
    keep = set(idx[top])
    log(f"   selected {len(keep)} CpGs; re-reading values ...")
    parts = []
    for ch in pd.read_csv(path, sep="\t", index_col=0, chunksize=chunk):
        ch = norm_cols(ch)
        sel = ch.index.intersection(keep)
        if len(sel):
            parts.append(ch.loc[sel, cols].astype(np.float32))
    met = pd.concat(parts)
    met.index.name = "cpg"
    met = met[~met.index.duplicated()]
    log(f"   methylation matrix {met.shape}")
    met.to_csv(out)
    return met


SILENT = {"Silent", "Intron", "IGR", "3'UTR", "5'UTR", "3'Flank", "5'Flank",
          "RNA", "lincRNA", "Splice_Region"}


def load_mutations(min_freq=0.02):
    out = os.path.join(PRC, "mut_matrix.csv")
    mafout = os.path.join(PRC, "mut_maf.parquet")
    if os.path.exists(out):
        return pd.read_csv(out, index_col=0)
    files = sorted(glob.glob(os.path.join(RAW, "gdc_maf", "*.maf.gz")))
    log(f"parsing {len(files)} MAF files ...")
    frames = []
    for i, f in enumerate(files):
        try:
            d = pd.read_csv(f, sep="\t", comment="#", low_memory=False,
                            compression="gzip",
                            usecols=["Hugo_Symbol", "Variant_Classification",
                                     "Tumor_Sample_Barcode", "HGVSp_Short",
                                     "Variant_Type", "Chromosome",
                                     "Start_Position", "End_Position"])
            frames.append(d)
        except Exception as e:  # noqa: BLE001
            log(f"   !! {os.path.basename(f)}: {e}")
        if (i + 1) % 200 == 0:
            log(f"   {i+1}/{len(files)} parsed")
    maf = pd.concat(frames, ignore_index=True)
    log(f"  total variant rows {len(maf)}")
    maf["Tumor_Sample_Barcode"] = maf["Tumor_Sample_Barcode"].str[:15]
    maf = maf[maf["Tumor_Sample_Barcode"].str[13:15].astype(int).between(1, 9)]
    maf = maf[~maf["Variant_Classification"].isin(SILENT)]
    maf.to_parquet(mafout, index=False)
    samples = sorted(maf["Tumor_Sample_Barcode"].unique())
    log(f"  non-silent variants {len(maf)} in {len(samples)} tumour samples")
    mat = (maf.assign(v=1)
              .pivot_table(index="Hugo_Symbol", columns="Tumor_Sample_Barcode",
                           values="v", aggfunc="max", fill_value=0))
    mat = mat.loc[mat.mean(axis=1) >= min_freq]
    log(f"  mutation matrix {mat.shape} (gene freq >= {min_freq})")
    mat.to_csv(out)
    return mat


def load_clinical():
    out = os.path.join(PRC, "clinical_tcga.csv")
    if os.path.exists(out):
        return pd.read_csv(out)
    cbp = os.path.join(RAW, "cbioportal")

    def widen(fn, key):
        d = json.load(open(os.path.join(cbp, fn)))
        d = pd.DataFrame(d)
        w = d.pivot_table(index=[key], columns="clinicalAttributeId",
                          values="value", aggfunc="first")
        w.index.name = key
        return w

    tcga = widen("brca_tcga_pan_can_atlas_2018_clinical_full.json", "patientId")
    smp = widen("brca_tcga_pan_can_atlas_2018_clinical_full.json", "sampleId")
    smp = smp.dropna(axis=1, how="all")
    log(f"  TCGA patient clinical {tcga.shape}; sample clinical {smp.shape}")
    tcga.to_csv(os.path.join(PRC, "tcga_clinical_patient.csv"))
    smp.to_csv(os.path.join(PRC, "tcga_clinical_sample.csv"))

    met = widen("brca_metabric_clinical_full.json", "patientId")
    mets = widen("brca_metabric_clinical_full.json", "sampleId")
    met.to_csv(os.path.join(PRC, "metabric_clinical_patient.csv"))
    mets.to_csv(os.path.join(PRC, "metabric_clinical_sample.csv"))
    log(f"  METABRIC patient clinical {met.shape}; sample clinical {mets.shape}")

    # survival file from Xena (OS in days, harmonised) for cross-check
    sv = pd.read_csv(os.path.join(RAW, "TCGA-BRCA.survival.tsv.gz"), sep="\t")
    sv.to_csv(os.path.join(PRC, "tcga_survival_xena.csv"), index=False)
    log(f"  Xena survival {sv.shape}")

    tcga.to_csv(out)
    return tcga


if __name__ == "__main__":
    t0 = time.time()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    ann = get_gene_annotation()
    if what in ("all", "expr"):
        load_expression(ann)
    if what in ("all", "mirna"):
        load_mirna()
    if what in ("all", "clinical"):
        load_clinical()
    if what in ("all", "mut"):
        load_mutations()
    if what in ("all", "methyl"):
        load_methylation()
    log(f"elapsed {time.time()-t0:.0f}s")
    with open(os.path.join(PROJ, "results", "preprocess_log.txt"), "a",
              encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
