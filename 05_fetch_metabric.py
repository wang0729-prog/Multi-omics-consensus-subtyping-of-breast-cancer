"""
05_fetch_metabric.py - pull METABRIC mRNA expression for a selected gene panel
from the cBioPortal REST API (the bulk datahub tarball is not reachable from
this network, so the data are assembled gene-by-gene).

Usage:  python 05_fetch_metabric.py <gene_list_file_or_comma_sep>
        python 05_fetch_metabric.py --all-from results/cluster/subtype_signature.csv
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

BASE = "https://www.cbioportal.org/api"
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "data", "processed")
os.makedirs(OUT, exist_ok=True)

STUDY = "brca_metabric"
PROFILE = "brca_metabric_mrna"
SAMPLE_LIST = "brca_metabric_all"


def post(path, payload, retries=5, timeout=600):
    body = json.dumps(payload).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(
                BASE + path, data=body,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"   ! {path} try{a+1}: {e}")
            time.sleep(4)
    return None


def symbols_to_entrez(symbols):
    """Batch symbol -> entrez id via cBioPortal /genes?keyword= (fallback)."""
    cache = os.path.join(OUT, "_gene_lookup.json")
    lut = {}
    if os.path.exists(cache):
        lut = json.load(open(cache))
    todo = [s for s in symbols if s not in lut]
    # bulk query: /genes returns all genes; use it once
    if not os.path.exists(os.path.join(OUT, "_all_genes.tsv")):
        print("  downloading full gene reference table ...")
        for kw in ["", "A", "B"]:
            pass
        req = urllib.request.Request(BASE + "/genes", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            genes = json.loads(r.read().decode())
        g = pd.DataFrame(genes)
        g.to_csv(os.path.join(OUT, "_all_genes.tsv"), sep="\t", index=False)
        print(f"   {g.shape} genes")
    g = pd.read_csv(os.path.join(OUT, "_all_genes.tsv"), sep="\t")
    g = g[g["type"] == "protein-coding"]
    m = dict(zip(g["hugoGeneSymbol"], g["entrezGeneId"]))
    for s in todo:
        if s in m:
            lut[s] = int(m[s])
    json.dump(lut, open(cache, "w"))
    return lut


def fetch_matrix(symbols, chunk=250):
    lut = symbols_to_entrez(symbols)
    found = [(s, lut[s]) for s in symbols if s in lut]
    print(f"  {len(found)}/{len(symbols)} symbols mapped to entrez ids")
    if not found:
        return pd.DataFrame()
    rows = []
    for i in range(0, len(found), chunk):
        part = found[i:i + chunk]
        ids = [e for _, e in part]
        d = post(f"/molecular-profiles/{PROFILE}/molecular-data/fetch",
                 {"entrezGeneIds": ids, "sampleListId": SAMPLE_LIST})
        if d:
            rows.extend(d)
        print(f"   chunk {i//chunk+1}: {len(d) if d else 0} records")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    e2s = {e: s for s, e in found}
    df["symbol"] = df["entrezGeneId"].map(e2s)
    df["sample"] = df["sampleId"].str[:15]
    mat = df.pivot_table(index="symbol", columns="sample", values="value",
                         aggfunc="mean")
    return mat


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--all-from"):
        src = sys.argv[2]
        s = pd.read_csv(src)
        # the signature table is one column per subtype -> collect every cell
        genes = sorted({str(v) for v in s.values.ravel()
                        if isinstance(v, str) and v and v != "nan"})
    elif len(sys.argv) > 1:
        genes = [g.strip() for g in sys.argv[1].replace(",", " ").split()]
    else:
        # 无参数时 sys.argv[1] 会抛 IndexError，栈信息对使用者毫无帮助
        sys.exit("用法: python scripts/05_fetch_metabric.py --all-from "
                 "results/fig1/subtype_signature.csv\n"
                 "  或: python scripts/05_fetch_metabric.py GENE1,GENE2,...")
    print(f"requesting {len(genes)} genes")
    m = fetch_matrix(genes)
    out = os.path.join(OUT, "metabric_expression_panel.csv")
    if os.path.exists(out):
        old = pd.read_csv(out, index_col=0)
        m = pd.concat([old, m])
        m = m[~m.index.duplicated()]
    m.to_csv(out)
    print(f"saved {m.shape} -> {out}")
