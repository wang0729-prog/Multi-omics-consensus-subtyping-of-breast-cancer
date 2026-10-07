"""
02b_fetch_clinical.py - fetch full clinical tables for TCGA-BRCA (PanCancer Atlas)
and METABRIC from the cBioPortal REST API, both PATIENT and SAMPLE level.
"""
import json
import os
import time
import urllib.request

BASE = "https://www.cbioportal.org/api"
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "data", "raw", "cbioportal")
os.makedirs(OUT, exist_ok=True)


# cBioPortal 的 WAF 拒掉 urllib 默认 UA（403），必须自带 User-Agent。
UA = {"User-Agent": "brca-multiomics/1.0 (Python-urllib)"}


def post(path, payload, retries=4):
    body = json.dumps(payload).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(
                BASE + path, data=body,
                headers=dict(UA, **{"Content-Type": "application/json"},
                             Accept="application/json"))
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"   ! {path} try{a+1}: {e}")
            time.sleep(3)
    return None


def get(path):
    req = urllib.request.Request(BASE + path,
                                 headers=dict(UA, Accept="application/json"))
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def run(study):
    attrs = get(f"/studies/{study}/clinical-attributes")
    patients = get(f"/studies/{study}/patients")
    samples = get(f"/studies/{study}/samples")
    pid = [p["patientId"] for p in patients]
    sid = [s["sampleId"] for s in samples]
    groups = {}
    for a in attrs:
        groups.setdefault("PATIENT" if a["patientAttribute"] else "SAMPLE", []).append(
            a["clinicalAttributeId"])
    recs = []
    for lvl, aids in groups.items():
        entity = pid if lvl == "PATIENT" else sid
        for i in range(0, len(aids), 200):
            chunk = aids[i:i + 200]
            r = post(f"/studies/{study}/clinical-data/fetch?clinicalDataType={lvl}",
                     {"attributeIds": chunk, "ids": entity})
            n = len(r) if r else 0
            recs.extend(r or [])
            print(f"   {study} {lvl}: {len(chunk)} attrs -> {n}")
    with open(os.path.join(OUT, f"{study}_clinical_full.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f)
    print(f"  {study}: total {len(recs)} clinical records")
    return recs


if __name__ == "__main__":
    for s in ["brca_tcga_pan_can_atlas_2018", "brca_metabric"]:
        print("===", s, "===")
        run(s)
