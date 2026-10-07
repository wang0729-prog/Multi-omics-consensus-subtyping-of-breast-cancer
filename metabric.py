"""
metabric.py - METABRIC mRNA expression and survival access, used for the
external (cross-platform) validation of the consensus subtypes and of the
prognostic signature.

The bulk datahub tarball is not reachable from this network, so expression is
assembled gene-by-gene through the cBioPortal REST API and cached in
data/processed/metabric_expression_panel.csv (shared with 05_fetch_metabric.py).
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRC = os.path.join(PROJ, "data", "processed")
BASE = "https://www.cbioportal.org/api"
STUDY = "brca_metabric"
PROFILE = "brca_metabric_mrna"
SAMPLE_LIST = "brca_metabric_all"
PANEL = os.path.join(PRC, "metabric_expression_panel.csv")

# cBioPortal 的 WAF 拒掉 urllib 默认 UA（403），必须自带 User-Agent。
UA = {"User-Agent": "brca-multiomics/1.0 (Python-urllib)"}


def _post(path, payload, retries=4, timeout=900):
    body = json.dumps(payload).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(
                BASE + path, data=body,
                headers=dict(UA, **{"Content-Type": "application/json"},
                             Accept="application/json"))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"   ! {path} try{a+1}: {e}", flush=True)
            time.sleep(3)
    return None


def _symbol_to_entrez():
    cache = os.path.join(PRC, "_gene_lookup.json")
    lut = json.load(open(cache)) if os.path.exists(cache) else {}
    g = pd.read_csv(os.path.join(PRC, "_all_genes.tsv"), sep="\t")
    g = g[g["type"] == "protein-coding"]
    m = dict(zip(g["hugoGeneSymbol"], g["entrezGeneId"]))
    lut.update({s: int(m[s]) for s in m})
    json.dump(lut, open(cache, "w"))
    return lut


def fetch_matrix(symbols, chunk=100):
    """Fetch a gene x sample expression matrix for the requested symbols."""
    lut = _symbol_to_entrez()
    found = [(s, lut[s]) for s in symbols if s in lut]
    print(f"  METABRIC: {len(found)}/{len(symbols)} symbols mapped", flush=True)
    rows = []
    for i in range(0, len(found), chunk):
        ids = [e for _, e in found[i:i + chunk]]
        d = _post(f"/molecular-profiles/{PROFILE}/molecular-data/fetch",
                  {"entrezGeneIds": ids, "sampleListId": SAMPLE_LIST})
        if d:
            rows.extend(d)
        print(f"   chunk {i//chunk+1}/{int(np.ceil(len(found)/chunk))}: "
              f"{len(d) if d else 0} records", flush=True)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    e2s = {e: s for s, e in found}
    df["symbol"] = df["entrezGeneId"].map(e2s)
    df["sample"] = df["sampleId"].str[:15]
    return df.pivot_table(index="symbol", columns="sample", values="value",
                          aggfunc="mean")


def ensure_expression(symbols):
    """Return a gene x sample matrix covering `symbols`, fetching what is new."""
    old = pd.read_csv(PANEL, index_col=0) if os.path.exists(PANEL) else pd.DataFrame()
    have = set(old.index)
    todo = [s for s in symbols if s not in have]
    if todo:
        print(f"  METABRIC: fetching {len(todo)} new genes", flush=True)
        new = fetch_matrix(todo)
        old = pd.concat([old, new]) if len(old) else new
        old = old[~old.index.duplicated()]
        old.to_csv(PANEL)
    return old.reindex([s for s in symbols if s in old.index])


def survival():
    """Patient-level overall survival for METABRIC."""
    c = pd.read_csv(os.path.join(PRC, "metabric_clinical_patient.csv"), index_col=0)
    c.index.name = "patient"
    t = pd.to_numeric(c["OS_MONTHS"], errors="coerce")
    e = c["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    d = pd.DataFrame({"os_time": t, "os_event": e})
    return d[d["os_time"].notna() & (d["os_time"] > 0)]
