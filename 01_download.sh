#!/usr/bin/env bash
# Download all raw data for the BRCA multi-omics project (resumable).
set -u
cd "$(dirname "$0")/.."
mkdir -p results data/raw || exit 1
mkdir -p results data/raw
RAW=data/raw
mkdir -p "$RAW"

GDC="https://gdc-hub.s3.us-east-1.amazonaws.com/download"
XENA="https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
LO="https://www.linkedomics.org/data_download/CPTAC-BRCA"

dl () {  # dl <url> <outfile>
  local url="$1" out="$2"
  echo "[$(date +%H:%M:%S)] GET $(basename "$out")"
  curl -sL --retry 3 --retry-delay 3 -C - -m 5400 -o "$out" "$url"
  echo "[$(date +%H:%M:%S)] -> $(basename "$out") $(stat -c%s "$out" 2>/dev/null) bytes"
}

# --- 1. TCGA-BRCA bulk multi-omics (GDC hub) ---
dl "$GDC/TCGA-BRCA.star_counts.tsv.gz"     "$RAW/TCGA-BRCA.star_counts.tsv.gz"     &
dl "$GDC/TCGA-BRCA.star_fpkm.tsv.gz"       "$RAW/TCGA-BRCA.star_fpkm.tsv.gz"       &
dl "$GDC/TCGA-BRCA.mirna.tsv.gz"           "$RAW/TCGA-BRCA.mirna.tsv.gz"           &
dl "$GDC/TCGA-BRCA.methylation450.tsv.gz"  "$RAW/TCGA-BRCA.methylation450.tsv.gz"  &
dl "$GDC/TCGA-BRCA.survival.tsv.gz"        "$RAW/TCGA-BRCA.survival.tsv.gz"        &
wait
echo "[$(date +%H:%M:%S)] bulk multi-omics DONE"

# --- 2. CNV (GISTIC thresholded, legacy Xena) ---
dl "$XENA/TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz" \
   "$RAW/TCGA.BRCA.Gistic2_CopyNumber_all_thresholded.by_genes.gz" &
dl "$XENA/TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz" "$RAW/TCGA.BRCA.HiSeqV2_PANCAN.gz" &
wait

# --- 3. CPTAC-BRCA proteome / phosphoproteome (LinkedOmics) ---
for f in HS_CPTAC_BRCA_2018_CLI.tsi \
         HS_CPTAC_BRCA_2018_Proteome_Ratio_Norm_gene_Median.cct \
         HS_CPTAC_BRCA_2018_Phosphoproteome_Ratio_Norm_Gene_median.cct \
         HS_CPTAC_BRCA_2018_Phosphoproteome_Ratio_Norm_Site.cct \
         HS_CPTAC_BRCA_2018_RNA_GENE.cct \
         HS_CPTAC_BRCA_2018_MUT_GENE.cbt ; do
  [ -s "$RAW/$f" ] || dl "$LO/$f" "$RAW/$f"
done

echo "[$(date +%H:%M:%S)] ALL DOWNLOADS FINISHED"
ls -la "$RAW"
