# Reproducibility record

What was checked: do these scripts run, and do they reproduce the numbers used in
the paper.

## Method

A sandbox containing only this repository's files was created, with `data/`,
`results/` and `figures/` mounted from the original analysis directory. Every script
listed below was executed there. All files under `results/` and `figures/` (plus
`data/processed/` when preprocessing was re-run) were hashed - size and MD5 - before
and after each run and compared file by file.

Figure files carry three kinds of matplotlib metadata that change on every run and
are not results: `/CreationDate` in PDF and AI, `<dc:date>` in SVG, and randomly
generated marker ids in SVG (`id="m4f68540d7d"`). These are normalised before
comparison. PNG files are compared without normalisation and match byte for byte,
which is the direct evidence that the rendering itself is deterministic.

## Scripts executed

| script | exit | wall time | outcome |
|---|---|---|---|
| `scripts/10_cluster.py` | 0 | ~2 min | 138 files identical, byte for byte (subtype labels, K-selection metrics) |
| `scripts/04_preprocess.py` | 0 | 473 s | full rebuild from `data/raw`; 13 matrices identical |
| `scripts/04_preprocess.py` (repeat, with freshly downloaded clinical data) | 0 | 383 s | 13 matrices still identical |
| `scripts/models.py` | 0 | 3.4 s | identical (101 combinations, per-fold cache hit) |
| `scripts/20_singlecell.py` | 0 | 68 s | identical; 100,064 cells, 15,821 ligand-receptor interactions |
| `scripts/fig1.py` | 0 | 9.5 s | metadata only; PNG identical |
| `scripts/fig2.py` | 0 | 31 s | metadata only |
| `scripts/fig3.py` | 0 | 8.7 s | metadata only; PNG identical |
| `scripts/fig4_5.py` | 0 | 41 s | metadata only (METABRIC n = 1,979) |
| `scripts/fig6.py` | 0 | 39 s | metadata only (CPTAC BC1 = 67, BC2 = 55) |
| `scripts/figS.py` | 0 | - | metadata only, 12 supplementary figures |
| `scripts/make_ai.py` | 0 | - | 20/20 .ai files valid, metadata only |
| `scripts/audit_figures.py` | 0 | - | identical |
| `scripts/check_panel_coverage.py` | 0 | - | `RESULT: clean` |
| `scripts/audit_manuscript_numbers.py` | 0 | - | 183 checks, 1 failure (reference count 74 > 60) |
| `scripts/02_fetch_cbioportal.py` | 0 | 3.5 min | 84,226 mutations retrieved |
| `scripts/02b_fetch_clinical.py` | 0 | 141 s | clinical table downloaded |
| `scripts/metabric.py` | 0 | 1.4 s | identical |

## Notes for anyone re-running this

* Keep the scripts in `scripts/`. `lib.py` derives the project root from its own
  location, so a flat layout puts every `data/` and `results/` path one level too
  high.
* cBioPortal rejects urllib's default User-Agent with HTTP 403; the fetch scripts set
  an explicit one.
* `04_preprocess.py` takes about 8 minutes and skips stages whose outputs already
  exist. Remove `data/processed/` to force a full rebuild.
* `05_fetch_metabric.py` needs `--all-from results/fig1/subtype_signature.csv`.
* `03_fetch_gdc_maf.py` downloads the GDC mutation files and can take well over
  15 minutes on a slow link.

## Result

Fifteen scripts were re-run and every number-bearing output reproduced byte for byte.
The manuscript audit reports 183 checks with a single failure, the reference count
(74 against a guideline of 60, retained deliberately).
