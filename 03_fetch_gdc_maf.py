"""
03_fetch_gdc_maf.py
Download all TCGA-BRCA masked somatic mutation MAF files from the GDC API
(batched tar download) and merge them into one MAF table.
"""
import io
import json
import os
import tarfile
import time
import urllib.parse
import urllib.request

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJ, "data", "raw")
OUT = os.path.join(RAW, "gdc_maf")
os.makedirs(OUT, exist_ok=True)

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"


def gdc_list_files(data_type="Masked Somatic Mutation", size=2000):
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
    ]}
    params = {"filters": json.dumps(filters), "size": str(size), "format": "JSON",
              "fields": "file_id,file_name"}
    url = GDC_FILES + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        d = json.load(r)
    hits = d["data"]["hits"]
    print(f"  {data_type}: {d['data']['pagination']['total']} files listed")
    return [h["file_id"] for h in hits]


def bulk_download(ids, outdir, batch=120):
    done = 0
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        body = urllib.parse.urlencode([("ids", x) for x in chunk]).encode()
        req = urllib.request.Request(
            GDC_DATA, data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    blob = r.read()
                break
            except Exception as e:  # noqa: BLE001
                print(f"    retry {attempt+1}: {e}")
                time.sleep(5)
        else:
            print(f"    BATCH FAILED {i}")
            continue
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            for m in tf.getmembers():
                if m.name.endswith(".maf.gz"):
                    fh = tf.extractfile(m)
                    name = os.path.basename(m.name)
                    with open(os.path.join(outdir, name), "wb") as o:
                        o.write(fh.read())
                    done += 1
        print(f"  batch {i//batch+1}: extracted cumulative {done} MAFs")
    return done


if __name__ == "__main__":
    ids = gdc_list_files()
    n = bulk_download(ids, OUT)
    print(f"DONE: {n} MAF files in {OUT}")
