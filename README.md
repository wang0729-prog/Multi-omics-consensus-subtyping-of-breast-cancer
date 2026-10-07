# Multi-omics consensus subtyping of breast cancer

Consensus clustering of five omics layers (mRNA, lncRNA, miRNA, DNA methylation and
somatic mutation) in 666 TCGA breast tumours, with proteomic, phosphoproteomic,
single-cell and METABRIC validation, and a 101-combination machine-learning
prognostic model.

Result: two consensus subtypes (BC1, n = 530; BC2, n = 136) that differ in
proliferative signalling, genomic instability and outcome.

## Data

All inputs are previously published public data; no primary data were generated.

| Source | Content | Retrieval |
|---|---|---|
| GDC data portal / UCSC Xena | TCGA-BRCA mRNA, lncRNA, miRNA, methylation, MAF, clinical | `03_fetch_gdc_maf.py`, `02_fetch_cbioportal.py` |
| cBioPortal | harmonised TCGA clinical + PAM50; METABRIC expression and clinical | `02_fetch_cbioportal.py`, `05_fetch_metabric.py` |
| LinkedOmics / Proteomic Data Commons | CPTAC-BRCA proteome and phosphoproteome | `04_preprocess.py` |
| CELLxGENE Census | breast cancer single-cell atlas (100,064 cells) | `20_singlecell.py` |

## Language and software

Everything here is Python; there is no R code, and no result in the paper was
produced by R. Clustering and machine learning use scikit-learn, survival modelling
uses lifelines and scikit-survival, enrichment uses gseapy, and the single-cell
atlas uses scanpy with liana for ligand-receptor inference.

R packages are cited in the paper for methodological provenance only: MOVICS for the
multi-algorithm consensus design, limma for the moderated t-statistic, maftools for
the oncoprint convention, ConsensusClusterPlus for consensus clustering. That
behaviour is reimplemented here in Python; the cited papers, not the cited packages,
define the method.

## Environment

Python 3.13 with the packages in `requirements.txt`. Verified on Windows; the scripts
use `os.path` throughout, so Linux and macOS need no change.

## Reproducing the analysis

From the repository root:

```bash
bash scripts/01_download.sh     # raw public downloads (GDC, UCSC Xena, cBioPortal)
bash scripts/run_all.sh         # whole pipeline; one log per stage in results/
```

or stage by stage:

```bash
python scripts/02_fetch_cbioportal.py     # TCGA clinical + PAM50 via cBioPortal
python scripts/02b_fetch_clinical.py      # clinical table
python scripts/03_fetch_gdc_maf.py        # somatic mutations from GDC
python scripts/04_preprocess.py           # build the five omics matrices
python scripts/05_fetch_metabric.py       # METABRIC expression and clinical
python scripts/metabric.py                # METABRIC nearest-template projection
python scripts/10_cluster.py              # 11 algorithms x 5 layers, K selection
python scripts/models.py                  # 101 machine-learning combinations
python scripts/20_singlecell.py           # single-cell atlas, ligand-receptor inference
python scripts/fig1.py                    # consensus + validation panels
python scripts/fig2.py fig3.py fig4_5.py fig6.py figS.py
```

`run_all.sh` calls `fig1.py` twice on purpose: the first pass writes the subtype
signature that `05_fetch_metabric.py` needs in order to project METABRIC samples.

Every `fig*.py` writes its own numbers to `results/<stage>/*.csv` and renders AI,
SVG, PDF and 300-dpi PNG. `make_ai.py` lifts the verified vector PDFs into Adobe
Illustrator files.

Layout note: the scripts live in `scripts/` and must stay there. `lib.py` derives the
project root from its own location (`dirname(dirname(__file__))`); moving the scripts
to the repository root makes every `data/` and `results/` path resolve one level too
high.

`audit_figures.py`, `check_panel_coverage.py` and `audit_manuscript_numbers.py`
re-check the figures and the numbers quoted in the manuscript.

## Outputs

* `figures/` - Fig1 to Fig8 and FigS1 to FigS12, as AI, SVG, PDF and 300-dpi PNG
* `results/` - every number quoted in the manuscript, as CSV
* `data/processed/` - the five-layer matrices consumed by the figure scripts

The intermediates under `data/processed` (500 MB) and `results/sc` (2.1 GB) are not
deposited in the repository; `04_preprocess.py` and `20_singlecell.py` regenerate
them, and `results/` without the single-cell intermediates is about 100 MB. Each
figure script reads only `results/<stage>/*.csv`, so any figure can be redrawn from
those tables alone once they exist.

## Quality control

```bash
python _qa/_static_check.py        # syntax, undefined globals, local paths, AI glyphs
python _qa/_repo_audit.py          # file inventory, private data, compliance
python _qa/_negative_control.py    # inject 5 defect types, confirm they are caught
```

All three must finish with zero problems. `audit_manuscript_numbers.py` re-derives
every number in the manuscript from `results/`; set `BRCA_MANUSCRIPT_DOCX` to the
manuscript `.docx` to run its formatting checks as well. The manuscript itself is not
part of this repository.

## Figure numbering

This repository keeps the original 12-supplementary-figure scheme. The journal version
of the paper reports nine supplementary figures; FigS7, FigS8, FigS11 and FigS12 here
correspond to supplementary figures 6, 7, 8 and 9 of the manuscript, and FigS6,
FigS9 and FigS10 are not used.

## Citation

Cite the article and the code repository with the DOI you will be given on deposit.
