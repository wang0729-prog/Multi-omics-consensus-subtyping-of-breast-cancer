"""
02_fetch_cbioportal.py
Fetch TCGA-BRCA (PanCancer Atlas) mutations + clinical/PAM50, and METABRIC
mRNA + clinical from the cBioPortal REST API (real public data).
Outputs go to data/raw/cbioportal/.
"""
import json
import os
import time
import urllib.request

BASE = "https://www.cbioportal.org/api"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cbioportal")
os.makedirs(OUT, exist_ok=True)

# cBioPortal 的 WAF 会拒掉 urllib 的默认 User-Agent（Python-urllib/3.x）返回 403，
# 而同样的 URL 用 curl 是 200。所有请求必须自带 UA，否则复现者会卡在这一步。
UA = {"User-Agent": "brca-multiomics/1.0 (Python-urllib)"}


def api(path, payload=None, method=None, retries=4):
    url = BASE + path
    for attempt in range(retries):
        try:
            if payload is None:
                headers = dict(UA, Accept="application/json")
                req = urllib.request.Request(url, headers=headers)
                m = method or "GET"
            else:
                data = json.dumps(payload).encode()
                headers = dict(UA, Accept="application/json",
                               **{"Content-Type": "application/json"})
                req = urllib.request.Request(
                    url, data=data, method=method or "POST", headers=headers)
                m = method or "POST"
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode("utf-8")
            return json.loads(body) if body.strip() else None
        except Exception as e:  # noqa: BLE001
            print(f"    ! {m} {path} attempt {attempt+1}: {e}")
            time.sleep(3)
    return None


def get_attributes(study):
    a = api(f"/studies/{study}/clinical-attributes") or []
    return a


def fetch_clinical(study):
    """Fetch all clinical attributes for a study via /studies/{id}/clinical-data/fetch."""
    attrs = get_attributes(study)
    if not attrs:
        print(f"  no clinical attributes for {study}")
        return None
    by_type = {}
    for a in attrs:
        dt = "PATIENT" if a.get("patientAttribute") else "SAMPLE"
        by_type.setdefault(dt, []).append(a["clinicalAttributeId"])
    all_records = []
    for dt, ids in by_type.items():
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            res = api(
                f"/studies/{study}/clinical-data/fetch?clinicalDataType={dt}",
                {"attributeIds": chunk, "ids": []})
            if res:
                all_records.extend(res)
            print(f"    {study} {dt}: {len(chunk)} attrs -> {len(res) if res else 0} records")
    return all_records


def fetch_all_patients(study):
    return api(f"/studies/{study}/patients") or []


def fetch_all_samples(study):
    return api(f"/studies/{study}/samples") or []


def fetch_sample_lists(study):
    return api(f"/studies/{study}/sample-lists") or []


def fetch_mutations(study, profile, sample_list_id):
    """Stream all mutations for a molecular profile."""
    ids = api(f"/sample-lists/{sample_list_id}/sample-ids") or []
    print(f"    {study}: {len(ids)} samples in {sample_list_id}")
    recs = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        # sampleListId 与 sampleIds 互斥，同时给会被 cBioPortal 判 400（整批返回空）。
        # 既然这里按 100 个样本分块拉，就只传 sampleIds。
        res = api(f"/molecular-profiles/{profile}/mutations/fetch",
                  {"sampleIds": chunk})
        if res:
            recs.extend(res)
        print(f"      mutations {i+len(chunk)}/{len(ids)} -> total {len(recs)}")
    return recs


def save(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    print(f"  saved {name} ({os.path.getsize(p)/1e6:.1f} MB, n={len(obj) if isinstance(obj,(list,dict)) else '-'})")


if __name__ == "__main__":
    print("=== sample lists ===")
    for s in ["brca_tcga_pan_can_atlas_2018", "brca_metabric"]:
        sl = fetch_sample_lists(s)
        print(s, "->", [(x["sampleListId"], x["name"], x.get("sampleCount")) for x in (sl or [])][:6])

    print("=== patients/samples ===")
    store = {}
    for s in ["brca_tcga_pan_can_atlas_2018", "brca_metabric"]:
        p = fetch_all_patients(s)
        sm = fetch_all_samples(s)
        store[s] = (p, sm)
        save(p, f"{s}_patients.json")
        save(sm, f"{s}_samples.json")

    print("=== clinical ===")
    for s in ["brca_tcga_pan_can_atlas_2018", "brca_metabric"]:
        # fetch_clinical() 自己按 study 拉全部临床属性，不需要 patients/samples；
        # 之前这里传了 3 个参数，直接 TypeError，这条路径从来没跑通过。
        c = fetch_clinical(s)
        save(c, f"{s}_clinical.json")

    print("=== mutations TCGA ===")
    mut = fetch_mutations("brca_tcga_pan_can_atlas_2018",
                          "brca_tcga_pan_can_atlas_2018_mutations",
                          "brca_tcga_pan_can_atlas_2018_all")
    save(mut, "brca_tcga_pan_can_atlas_2018_mutations.json")

    print("DONE")
