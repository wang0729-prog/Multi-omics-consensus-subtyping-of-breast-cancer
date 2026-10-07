"""Independent audit of every number in the npj PO manuscript.

Design rules (from the manuscript-data-audit skill):
  * recompute from data/processed (raw matrices) wherever the computation is
    affordable; use results/*.csv only for quantities that cannot be
    recomputed here (ML cross-validation folds, liana permutations)
  * every check prints [ok] / [FAIL] item value
  * a final summary reports checks run / failures

Run:  python scripts/audit_manuscript_numbers.py
Out:  results/qa/manuscript_audit_<date>.txt
"""
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

VERBOSE = '--quiet' not in sys.argv
CHECKS = []
FAILS = []
NOTES = []


def chk(section, item, expected, got, tol=0.0, note=''):
    """expected: what the manuscript says. got: what the data gives."""
    def eq(a, b):
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        try:
            fa, fb = float(a), float(b)
            if np.isnan(fa) and np.isnan(fb):
                return True
            return abs(fa - fb) <= tol + 1e-12
        except (TypeError, ValueError):
            return str(a) == str(b)

    ok = eq(expected, got)

    def pretty(v):
        if isinstance(v, (tuple, list)):
            return "(" + ", ".join(pretty(x) for x in v) + ")"
        if hasattr(v, 'item'):
            try:
                return str(v.item())
            except Exception:
                pass
        return str(v)

    CHECKS.append((section, item, ok))
    if not ok:
        FAILS.append((section, item, pretty(expected), pretty(got), note))
        print(f'  [FAIL] {item}\n         manuscript = {pretty(expected)}\n'
              f'         data       = {pretty(got)}{"   <-- " + note if note else ""}')
    elif VERBOSE:
        print(f'  [ok]   {item}  = {pretty(got)}')
    return ok


def pct(x, n):
    return round(100 * x / n, 0)


def chk_p(section, item, expected, got, note=''):
    """Compare p-values the way the manuscript reports them: 2 significant figures."""
    ok = (abs(expected - got) <= 0.05 * abs(expected)) if expected else (got == 0)
    CHECKS.append((section, item, ok))
    if not ok:
        FAILS.append((section, item, f'{expected:.3g}', f'{got:.3g}', note))
        print(f'  [FAIL] {item}\n         manuscript = {expected:.3g}\n'
              f'         data       = {got:.3g}{"   <-- " + note if note else ""}')
    elif VERBOSE:
        print(f'  [ok]   {item}  = {got:.3g}')
    return ok


def main():
    print('=' * 78)
    print('MANUSCRIPT NUMBER AUDIT - npj Precision Oncology submission')
    print('=' * 78)

    sub = pd.read_csv('results/cluster/subtypes.csv')
    sub['sample'] = sub['sample'].str[:15]
    sub['patient'] = sub['sample'].str[:12]
    clin = pd.read_csv('data/processed/tcga_clinical_patient.csv')
    clin['patient'] = clin['patientId'].str[:12]

    # ============================================================ 1  cohort
    print('\n-- 1. Cohort and feature counts -------------------------------')
    chk('1', 'clustering cohort size', 666, len(sub))
    chk('1', 'BC1 n', 530, int((sub.cluster == 'BC1').sum()))
    chk('1', 'BC2 n', 136, int((sub.cluster == 'BC2').sum()))

    for f, label, exp in [('data/processed/mrna.csv', 'mRNA samples', 1102),
                          ('data/processed/lncrna.csv', 'lncRNA samples', 1102),
                          ('data/processed/mirna.csv', 'miRNA samples', 1085),
                          ('data/processed/methyl_top10000.csv', 'methylation samples', 789),
                          ('data/processed/mut_matrix.csv', 'mutation samples', 973)]:
        with open(f) as fh:
            n = len(fh.readline().rstrip('\n').split(',')) - 1
        chk('1', label, exp, n)

    mrna_genes = sum(1 for _ in open('data/processed/mrna.csv')) - 1
    lnc_genes = sum(1 for _ in open('data/processed/lncrna.csv')) - 1
    mir_genes = sum(1 for _ in open('data/processed/mirna.csv')) - 1
    mut_genes = sum(1 for _ in open('data/processed/mut_matrix.csv')) - 1
    chk('1', 'mRNA genes', 19212, mrna_genes, note='rows of mrna.csv')
    chk('1', 'lncRNA genes', 6507, lnc_genes)
    chk('1', 'miRNA features', 1881, mir_genes)
    chk('1', 'mutation panel genes (>=2%)', 166, mut_genes)

    # ==================================================== 2  K selection
    print('\n-- 2. Cluster-number selection --------------------------------')
    ks = pd.read_csv('results/cluster/k_selection.csv').set_index('K')
    chk('2', 'silhouette at K=2', 0.677, ks.loc[2, 'silhouette'], tol=5e-4)
    chk('2', 'PAC at K=2', 0.847, ks.loc[2, 'PAC'], tol=5e-4)
    chk('2', 'delta area at K=2', 0.598, ks.loc[2, 'delta_area'], tol=5e-4)
    chk('2', 'rank sum K=2', 3, int(ks.loc[2, 'rank_sum']))
    for k in (3, 4, 5, 6):
        chk('2', f'rank sum K={k}', {3: 6, 4: 9, 5: 12, 6: 15}[k], int(ks.loc[k, 'rank_sum']))
    lo, hi = ks.loc[3:6, 'PAC'].min(), ks.loc[3:6, 'PAC'].max()
    chk('2', 'PAC K=3-6 min', 0.978, lo, tol=5e-4)
    chk('2', 'PAC K=3-6 max', 0.995, hi, tol=5e-4)

    # ============================================== 3  PAM50 / histology
    print('\n-- 3. PAM50, histology, demographics --------------------------')
    ct = pd.read_csv('results/fig1/pam50_crosstab.csv', index_col=0)
    ct.index = [str(i) for i in ct.index]
    from scipy.stats import chi2_contingency, mannwhitneyu
    chi2, p, dof, _ = chi2_contingency(ct.values)
    chk('3', 'PAM50 chi2 p', 6.5e-115, p, tol=0.05e-115)
    if 'Basal' in ct.index:
        chk('3', 'basal-like in BC2', 106, int(ct.loc['Basal', 'BC2']))
        chk('3', 'basal-like in BC1', 1, int(ct.loc['Basal', 'BC1']))
        chk('3', 'basal-like total', 107, int(ct.loc['Basal'].sum()))
    for lab, exp1 in [('LumA', 335), ('LumB', 130), ('Her2', 24)]:
        if lab in ct.index:
            chk('3', f'BC1 {lab}', exp1, int(ct.loc[lab, 'BC1']))
    if 'Her2' in ct.index:
        chk('3', 'BC2 Her2', 14, int(ct.loc['Her2', 'BC2']))
    norm_like = [i for i in ct.index if 'ormal' in i]
    if norm_like:
        chk('3', 'BC2 normal-like', 12, int(ct.loc[norm_like[0], 'BC2']))
        # discussion statement: BC1 carried 15 normal-like
        chk('3', 'BC1 normal-like', 15, int(ct.loc[norm_like[0], 'BC1']))
    chk('3', 'BC2 LumA+LumB', 1, int(ct.loc[['LumA', 'LumB'], 'BC2'].sum()))
    NOTES.append(f'PAM50 crosstab covers {int(ct.values.sum())} of 666 tumours; '
                 f'23 without a PAM50 call (row "nan"), '
                 f'{666 - int(ct.values.sum())} tumours absent from the table')

    # histology from raw clinical (TUMOR_TYPE, as used by the pipeline)
    m = sub.merge(clin, on='patient', how='left')
    s2c = dict(zip(sub['sample'], sub['cluster']))
    NOTES.append(f'clinical annotation matched for {m["TUMOR_TYPE"].notna().sum()} of {len(m)} '
                 f'tumours in the clustering cohort')
    mapping = {'Infiltrating Ductal Carcinoma': 'Ductal',
               'Infiltrating Lobular Carcinoma': 'Lobular'}
    m['h'] = m['TUMOR_TYPE'].map(mapping).fillna('Other')
    hist = (m.groupby(['cluster', 'h']).size().unstack(fill_value=0)
             .reindex(['BC1', 'BC2'])[['Ductal', 'Lobular', 'Other']])
    print('         recomputed histology table (Ductal/Lobular/Other):')
    print(hist.to_string().replace('\n', '\n         '))
    frac = hist.div(hist.sum(axis=1), axis=0)
    chk('3', 'lobular % in BC1', 23, round(100 * frac.loc['BC1', 'Lobular']), tol=1.0)
    chk('3', 'lobular % in BC2', 2, round(100 * frac.loc['BC2', 'Lobular']), tol=1.0)
    _, ph, _, _ = chi2_contingency(hist.values)
    chk('3', 'histology chi2 p (2x3 table)', 1.2e-7, ph, tol=0.05e-7)

    # ==================================================== 4  survival
    print('\n-- 4. Survival endpoints -------------------------------------')
    se = pd.read_csv('results/supp/subtype_survival_endpoints.csv')
    for _, r in se.iterrows():
        print(f'         raw: {r.endpoint}  n={r.n} events={r.events} '
              f'logrank_p={r.logrank_p:.6g} HR={r.HR:.6g} ({r.lower:.4g}-{r.upper:.4g}) HRp={r.HR_p:.6g}')
    d = se.set_index('endpoint')
    for ep, exp_p in [('DFS', 0.0082), ('PFS', 0.051), ('DSS', 0.069), ('OS', 0.29)]:
        if ep in d.index:
            chk('4', f'{ep} log-rank p', exp_p, d.loc[ep, 'logrank_p'], tol=5e-4)
        else:
            chk('4', f'{ep} present in results', 'present', 'MISSING')
    if 'DFS' in d.index:
        chk('4', 'DFS HR', 2.18, d.loc['DFS', 'HR'], tol=5e-3)
        chk('4', 'DFS HR CI lower', 1.21, d.loc['DFS', 'lower'], tol=5e-3)
        chk('4', 'DFS HR CI upper', 3.95, d.loc['DFS', 'upper'], tol=5e-3)
        chk('4', 'DFS HR p', 0.0099, d.loc['DFS', 'HR_p'], tol=5e-5)
        chk('4', 'DFS n', 571, int(d.loc['DFS', 'n']), note='BC1 451 + BC2 120')
    if 'OS' in d.index:
        chk('4', 'OS n', 660, int(d.loc['OS', 'n']))
        chk('4', 'OS events', 85, int(d.loc['OS', 'events']))

    # KM group sizes recomputed from clinical + subtype
    dfa = m[m['DFS_MONTHS'].notna() & m['DFS_STATUS'].notna() & (m['DFS_MONTHS'] > 0)]
    chk('4', 'KM DFS BC1 n', 451, int((dfa.cluster == 'BC1').sum()), tol=0)
    chk('4', 'KM DFS BC2 n', 120, int((dfa.cluster == 'BC2').sum()), tol=0)

    # ================================================ 5  METABRIC NTP
    print('\n-- 5. METABRIC nearest-template prediction --------------------')
    ntp = pd.read_csv('results/fig1/metabric_ntp.csv')
    chk('5', 'METABRIC tumours attempted', 1980, len(ntp))
    passed = ntp[ntp['pass'].astype(str).str.lower().isin(['true', '1'])]
    chk('5', 'NTP passed', 1550, len(passed))
    chk('5', 'NTP passed %', 78, pct(len(passed), len(ntp)), tol=1.0)
    chk('5', 'NTP BC1', 866, int((passed.pred == 'BC1').sum()))
    chk('5', 'NTP BC2', 684, int((passed.pred == 'BC2').sum()))
    chk('5', 'METABRIC RFS log-rank p', 0.284, pd.read_csv('results/fig1/metabric_km_p.csv').iloc[0, 0], tol=5e-4)

    # ================================================ 6  mutation / TMB
    print('\n-- 6. Mutational landscape, TMB, FGA --------------------------')
    mf = pd.read_csv('results/fig2/mutation_frequency.csv')
    mf['freq'] = mf['freq'].astype(float)
    f_ok = mf.freq <= 1.0
    mfp = mf[f_ok].set_index('gene')['freq'] if f_ok.all() else mf.set_index('gene')['freq']
    for g, exp in [('PIK3CA', 33.8), ('TP53', 32.4), ('TTN', 16.8), ('CDH1', 15.8), ('GATA3', 11.9)]:
        if g in mfp.index:
            val = mfp[g] * (100 if mfp[g] <= 1 else 1)
            chk('6', f'{g} mutation %', exp, val, tol=0.05)
        else:
            chk('6', f'{g} in mutation_frequency.csv', 'present', 'MISSING')

    fga = m.set_index('sample')['FRACTION_GENOME_ALTERED'].astype(float)
    fga1 = fga[m.set_index('sample').cluster == 'BC1'].median()
    fga2 = fga[m.set_index('sample').cluster == 'BC2'].median()
    chk('6', 'FGA median BC1', 0.217, fga1, tol=5e-4)
    chk('6', 'FGA median BC2', 0.409, fga2, tol=5e-4)
    g1 = fga[m.set_index('sample').cluster == 'BC1'].dropna()
    g2 = fga[m.set_index('sample').cluster == 'BC2'].dropna()
    _, pf = mannwhitneyu(g1, g2, alternative='two-sided')
    NOTES.append(f'FGA Mann-Whitney p recomputed = {pf:.3g} (manuscript: p < 0.001); '
                 f'n = {len(g1)} BC1 + {len(g2)} BC2 = {len(g1) + len(g2)} of 666')

    # --- TMB: non-synonymous variant count per Mb (30 Mb callable exome).
    #     The MAF is pre-filtered to protein-altering classes, so variant-level
    #     counts ARE non-synonymous.  Fig. 2B and Methods use this definition.
    maf = pd.read_parquet('data/processed/mut_maf.parquet')
    maf = maf[maf['Tumor_Sample_Barcode'].str[:15].isin(set(s2c))]
    cnt = maf.groupby('Tumor_Sample_Barcode').size()
    tmb = cnt / 30.0
    g = pd.Series({k[:15]: s2c.get(k[:15]) for k in cnt.index})
    t1 = tmb[g == 'BC1']
    t2 = tmb[g == 'BC2']
    _, pT = mannwhitneyu(t1, t2, alternative='two-sided')
    print(f'         TMB (non-synonymous variant count / 30 Mb): '
          f'BC1 median {t1.median():.3f} (n={len(t1)}), BC2 median {t2.median():.3f} (n={len(t2)}), p = {pT:.3g}')

    chk('6', 'TMB median BC2', 1.80, t2.median(), tol=0.01)
    chk('6', 'TMB median BC1', 0.93, t1.median(), tol=0.01)
    chk('6', 'TMB Mann-Whitney p', 7.1e-12, pT, tol=0.1e-12)

    # ============================================ 7  DE and GSEA
    print('\n-- 7. Differential expression and GSEA -----------------------')
    deg = pd.read_csv('results/fig2/DEG_contrast.csv')
    up1 = ((deg.log2FC > 1) & (deg.FDR < 0.05)).sum()
    up2 = ((deg.log2FC < -1) & (deg.FDR < 0.05)).sum()
    chk('7', 'BC1-upregulated genes', 1520, int(up1))
    chk('7', 'BC2-upregulated genes', 1395, int(up2))
    gs = pd.read_csv('results/fig2/GSEA_hallmark.csv')
    gmap = gs.set_index('Term')['NES']
    for term, exp in [('E2F Targets', -2.66), ('G2-M Checkpoint', -2.47),
                      ('Estrogen Response Early', 2.58), ('Estrogen Response Late', 2.20)]:
        hit = [t for t in gmap.index
               if t.lower().replace('-', '').replace(' ', '') ==
               term.lower().replace('-', '').replace(' ', '')]
        if hit:
            chk('7', f'GSEA NES {term}', exp, gmap[hit[0]], tol=5e-3)
        else:
            chk('7', f'GSEA term {term} present', 'present', 'MISSING')

    # ==================================================== 8  immune
    print('\n-- 8. Immune / stromal composition ---------------------------')
    imm = pd.read_csv('results/fig3/immune_ssgsea.csv').set_index('Term')
    imm.columns = [c[:15] for c in imm.columns]
    grp = sub.set_index('sample').cluster
    common = [c for c in imm.columns if c in grp.index]
    imm = imm[common]
    g = grp.loc[common]
    rows = []
    for term in imm.index:
        v1 = imm.loc[term, (g == 'BC1').values].astype(float)
        v2 = imm.loc[term, (g == 'BC2').values].astype(float)
        _, pp = mannwhitneyu(v1, v2, alternative='two-sided')
        rows.append(dict(term=term, BC1=v1.mean(), BC2=v2.mean(), p=pp,
                         higher='BC2' if v2.mean() > v1.mean() else 'BC1'))
    ib = pd.DataFrame(rows)
    ib.to_csv('results/qa/immune_recomputed.csv', index=False)
    print('         recomputed per-population means written to results/qa/immune_recomputed.csv')

    # manuscript claims: population -> (p, pair as written, subtype quoted first)
    # 正文的写法是"句首主语在前"：BC2 showed higher ... (BC2 value versus BC1 value)，
    # 主语已经点名，所以 BC2-first 不是错误。之前按固定 BC1-first 判，属于假警报。
    ORDER_LOG = []
    claims = {
        'CD8_T_cells': (8.3e-17, None, None),
        'Cytotoxic_cells': (1.3e-11, (-0.043, -0.087), 'BC2'),
        'Treg': (1.3e-11, None, None),
        'NKT': (1.9e-10, None, None),
        'B_cells': (1.3e-7, None, None),
        'NK_cells': (6.2e-8, None, None),
        'Monocytes': (5.3e-33, (0.278, 0.161), 'BC2'),
        'Macrophages': (1.5e-8, None, None),
        'Mast_cells': (1.3e-46, (0.123, -0.140), 'BC1'),
        'Th2_cells': (3.8e-53, None, None),
        'Th17_cells': (4.5e-24, None, None),
        'Endothelial': (2.0e-12, None, None),
        'Pericytes': (1.8e-9, None, None),
    }
    ibi = ib.set_index('term')
    for term, (exp_p, exp_vals, exp_order) in claims.items():
        hit = [t for t in ibi.index if t.lower().replace('_', '').replace(' ', '') ==
               term.lower().replace('_', '')]
        if not hit:
            chk('8', f'population {term} present', 'present', 'MISSING')
            continue
        r = ibi.loc[hit[0]]
        chk_p('8', f'{hit[0]} p', exp_p, r.p, note=f'data: BC1 {r.BC1:.4g} / BC2 {r.BC2:.4g}')
        if exp_vals:
            got12 = (round(r.BC1, 3), round(r.BC2, 3))
            got21 = (round(r.BC2, 3), round(r.BC1, 3))
            def match_ok(exp, got):
                return all(abs(a - b) <= 5e-3 for a, b in zip(exp, got))
            got = got12 if exp_order == 'BC1' else got21
            ORDER_LOG.append((hit[0], exp_order + '-first'))
            chk('8', f'{hit[0]} values ({exp_order}-first)', exp_vals, got, tol=5e-3)

    ss = pd.read_csv('results/fig3/subtype_summary_stats.csv').set_index('feature')
    for feat, exp_p, exp_pair, tol in [
            ('Checkpoint_score', 4.0e-9, (7.36, 8.21), 5e-3),
            ('Cytolytic', 1.0e-5, (7.60, 8.30), 5e-3),
            ('Hypoxia', 8.2e-36, (-11.4, 10.4), 0.05),   # reported to 1 d.p.
            ('Immune_score', 1.2e-5, (7.72, 8.36), 5e-3),
            ('Stromal', 1.4e-5, (14.59, 14.12), 5e-3)]:
        if feat in ss.index:
            r = ss.loc[feat]
            got12 = (round(float(r.BC1_mean), 2), round(float(r.BC2_mean), 2))
            got21 = (got12[1], got12[0])
            # 这里的 exp_pair 是数据按 BC1-first 排好的一对；正文若以 BC2 作主语写成
            # BC2-first 是对的（主语已点名），因此两种顺序都接受。图注里没有主语，
            # 顺序由第 13 节单独按 BC1-first 约定检查。
            # 选顺序时的容差必须和 chk() 内部一致（tol + 1e-12），
            # 否则会出现"其实对得上、却选了反序"的假失败。
            def _ok(g):
                return all(abs(a - b) <= tol + 1e-12 for a, b in zip(exp_pair, g))
            got = got12 if _ok(got12) else (got21 if _ok(got21) else got12)
            chk('8', f'{feat} values', exp_pair, got, tol=tol)
            chk_p('8', f'{feat} p', exp_p, float(r.p))
        else:
            chk('8', f'{feat} present', 'present', 'MISSING')
    print('         value-pair ordering used in the manuscript:')
    for k, v in ORDER_LOG:
        print(f'           {k:<22}{v}')

    # ==================================================== 9  model
    print('\n-- 9. Machine-learning prognostic model ----------------------')
    sc = pd.read_csv('results/model/model_scores_101.csv')
    chk('9', 'number of combinations', 101, len(sc))
    chk('9', 'best combination is SuperPC', 'SuperPC', sc.loc[sc.mean_C.idxmax(), 'model'])
    chk('9', 'best mean C-index', 0.630, sc.mean_C.max(), tol=5e-4)
    chk('9', 'all combinations fitted', 101, int((sc.n_ok == 10).sum()))
    chk('9', 'genes in forest plot', 30, len(pd.read_csv('results/model/model_forest.csv')))
    rt = pd.read_csv('results/model/risk_table.csv')
    chk('9', 'model cohort n', 1075, len(rt))
    bench = pd.read_csv('results/model/benchmark.csv').set_index('predictor')['C_index']
    print('         raw benchmark:', {k: round(v, 4) for k, v in bench.items()})
    for name, exp in [('Stage', 0.689), ('This model', 0.646), ('Age', 0.634),
                      ('PAM50', 0.583)]:
        if name in bench.index:
            chk('9', f'C-index {name}', exp, bench[name], tol=5e-3)
        else:
            chk('9', f'benchmark predictor {name}', 'present', 'MISSING')
    ext = pd.read_csv('results/model/external_metabric_risk.csv')
    chk('9', 'METABRIC external n', 1979, len(ext))

    # ==================================================== 10  CPTAC
    print('\n-- 10. CPTAC proteome / phosphoproteome ----------------------')
    cpt = pd.read_csv('results/fig6/cptac_ntp_labels.csv')
    chk('10', 'CPTAC tumours', 122, len(cpt))
    chk('10', 'CPTAC BC1', 67, int((cpt.subtype == 'BC1').sum()))
    chk('10', 'CPTAC BC2', 55, int((cpt.subtype == 'BC2').sum()))
    chk('10', 'CPTAC median NTP margin', 0.112, cpt.margin.median(), tol=5e-4)
    cor = pd.read_csv('results/fig6/mrna_protein_correlation.csv')
    chk('10', 'signature genes measured in both layers', 223, len(cor))
    chk('10', 'median mRNA-protein rho', 0.65, cor.rho.median(), tol=5e-3)
    chk('10', 'pct genes rho > 0.3', 85.7, round(100 * (cor.rho > 0.3).mean(), 1), tol=0.1)
    cm = cor.set_index('gene')['rho']
    for g, exp in [('ESR1', 0.89), ('AGR3', 0.87)]:
        if g in cm.index:
            chk('10', f'{g} rho', exp, cm[g], tol=5e-3)
    dp = pd.read_csv('results/fig6/differential_proteins.csv')
    chk('10', 'quantified proteins', 12284, len(dp))
    sigp = dp[(dp.log2FC.abs() > 0.3) & (dp.FDR < 0.05)]
    chk('10', 'differential proteins', 2741, len(sigp))
    chk('10', 'differential proteins BC1-up', 1230, int((sigp.log2FC > 0).sum()))
    chk('10', 'differential proteins BC2-up', 1511, int((sigp.log2FC < 0).sum()))
    dps = pd.read_csv('results/fig6/differential_phosphosites.csv')
    chk('10', 'quantified phosphosites', 29776, len(dps))
    sigs = dps[(dps.log2FC.abs() > 0.4) & (dps.FDR < 0.05)]
    chk('10', 'differential phosphosites', 4274, len(sigs))
    chk('10', 'differential phosphosites BC1-up', 1818, int((sigs.log2FC > 0).sum()))
    chk('10', 'differential phosphosites BC2-up', 2456, int((sigs.log2FC < 0).sum()))
    lfc = dp.set_index('gene')['log2FC']
    for g, exp in [('ESR1', 3.64), ('MAPT', 3.75), ('TFF1', 3.85), ('DNAJC12', 3.87),
                   ('GFRA1', 3.88), ('SMOX', -8.08), ('FDCSP', -6.05), ('SLCO1B7', -5.64),
                   ('PPP1R14C', -4.80), ('S100A7A', -4.30), ('OLAH', -4.06)]:
        if g in lfc.index:
            chk('10', f'protein log2FC {g}', exp, lfc[g], tol=5e-3)
        else:
            chk('10', f'protein {g} present', 'present', 'MISSING')

    # ================================================= 11  single cell
    print('\n-- 11. Single-cell mapping -----------------------------------')
    meta = None
    if os.path.exists('results/sc/sc_processed.h5ad'):
        try:
            import anndata as ad
            meta = ad.read_h5ad('results/sc/sc_processed.h5ad', backed='r').obs
        except Exception as e:
            NOTES.append(f'could not open h5ad: {e}')
    if meta is not None:
        chk('11', 'total cells', 100064, meta.shape[0])
        chk('11', 'donors (tumours)', 26, meta['donor_id'].nunique())
        chk('11', 'major cell types', 9, meta['celltype_major'].nunique())
        chk('11', 'malignant (Cancer Epithelial) cells', 24489,
            int((meta['celltype_major'] == 'Cancer Epithelial').sum()))
        chk('11', 'normal epithelial cells', 4355,
            int((meta['celltype_major'] == 'Normal Epithelial').sum()))
        call = meta['normal_cell_call'].astype(str)
        chk('11', 'cells with no inferCNV call', 71220,
            int((call == 'no_inferCNV_call').sum()))
        chk('11', 'inferCNV cancer cells', 24489, int((call == 'cancer').sum()))
        chk('11', 'inferCNV normal cells', 4355, int((call == 'normal').sum()))
        if 'sc_subtype' in meta.columns:
            s = meta['sc_subtype'].astype(str)
            chk('11', 'BC1-like malignant cells', 12218, int((s == 'BC1').sum()))
            chk('11', 'BC2-like malignant cells', 12271, int((s == 'BC2').sum()))
            print('         malignant-cell subtype counts (recomputed):',
                  s.value_counts().to_dict())
    pw = pd.read_csv('results/sc/pathway_by_sc_subtype.csv')
    pwi = pw.set_index('programme')
    # manuscript convention: the HIGHER-scoring subtype is quoted first
    claims = {'pw_E2F Targets': (0.159, 0.018), 'pw_Glycolysis': (0.246, 0.198),
              'pw_OXPHOS': (0.502, 0.448), 'pw_PI3K-AKT-mTOR': (0.246, 0.216),
              'pw_EMT': (0.058, 0.027), 'pw_Inflammatory Response': (0.026, 0.024),
              'pw_Hypoxia': (0.211, 0.205), 'pw_TNF-alpha via NF-kB': (0.246, 0.177),
              'mb_Fatty Acid': (0.251, 0.246)}
    for prog, exp in claims.items():
        hit = [t for t in pwi.index if t.lower().replace('-', '').replace(' ', '') ==
               prog.lower().replace('-', '').replace(' ', '').replace('pw', 'pw')]
        if not hit:
            hit = [t for t in pwi.index if prog.split('_', 1)[-1].lower().replace('-', '')
                   in t.lower().replace('-', '').replace(' ', '')]
        if hit:
            r = pwi.loc[hit[0]]
            hi = max(r.BC1, r.BC2)
            lo = min(r.BC1, r.BC2)
            chk('11', f'{hit[0]} (higher-first)', exp, (round(hi, 3), round(lo, 3)), tol=5e-3,
                note=f'BC1={r.BC1:.3f} BC2={r.BC2:.3f} -> higher is '
                     f'{"BC2" if r.BC2 > r.BC1 else "BC1"}')
        else:
            chk('11', f'{prog} present', 'present', 'MISSING')
    print('         raw single-cell programmes:')
    for _, r in pw.iterrows():
        print(f'           {r.programme:<40} BC1={r.BC1:.4f} BC2={r.BC2:.4f} p={r.p:.4g}')
    lfam = pd.read_csv('results/sc/lr_families.csv')
    lf = lfam.set_index('family')
    for fam, cell, exp in [('MIF', 'Cancer Epithelial', 0.843),
                           ('MIF', 'Normal Epithelial', 0.825),
                           ('MIF', 'Plasmablasts', 0.803),
                           ('MHC', 'B-cells', 0.892), ('MHC', 'Myeloid', 0.887),
                           ('MHC', 'Cancer Epithelial', 0.659),
                           ('CXCL', 'CAFs', 0.586), ('FN1', 'CAFs', 0.669),
                           ('TGFB', 'Myeloid', 0.437)]:
        if fam in lf.index and cell in lf.columns:
            chk('11', f'liana {fam} from {cell}', exp, float(lf.loc[fam, cell]), tol=5e-4)
        else:
            chk('11', f'liana {fam}/{cell} present', 'present', 'MISSING')
    if os.path.exists('results/sc/liana_all.csv'):
        la = pd.read_csv('results/sc/liana_all.csv')
        chk('11', 'liana interactions reported', 15821, len(la), note='rows in liana_all.csv')
    print('         raw liana family weights:')
    for _, r in lfam.iterrows():
        vals = {k: round(float(v), 3) for k, v in r.items() if k != 'family' and pd.notna(v)}
        print(f'           {r.family:<8} {vals}')

    # ================================================ 12  manuscript QC
    print('\n-- 12. npj PO formatting requirements ------------------------')
    PS = []
    # 优先级：环境变量 > 投稿包里的当前版 > 工作稿 > 仓库根目录默认名。
    # 投稿包里那份是用户在 Word 里改过的当前版，必须优先于文章撰写/ 下的旧版。
    MANUSCRIPT = os.environ.get('BRCA_MANUSCRIPT_DOCX')
    if not MANUSCRIPT:
        for cand in ('npjPO_submission/01_manuscript/manuscript_npjPO_v1.docx',
                     '文章撰写/manuscript_npjPO_v1.docx',
                     'manuscript.docx'):
            if os.path.exists(cand):
                MANUSCRIPT = cand
                break
    if not MANUSCRIPT:
        MANUSCRIPT = 'manuscript.docx'
    print(f'         manuscript under test: {MANUSCRIPT}')
    if not os.path.exists(MANUSCRIPT):
        print('         [skip] manuscript .docx not found; the manuscript is not part of the '
              'public repository. Set BRCA_MANUSCRIPT_DOCX=/path/to/manuscript.docx to run '
              'this section.')
    else:
        from docx import Document
        doc = Document(MANUSCRIPT)
        ps = [p.text for p in doc.paragraphs]
        PS = ps

        def words(t):
            import re
            return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-\u2013\u2212.,%×⁰¹²³⁴⁵⁶⁷⁸⁹⁻]*", t))

        ai = ps.index('Abstract')
        ii = ps.index('Introduction')
        chk('12', 'abstract word count <= 150', True, words(ps[ai + 1]) <= 150,
            note=f'actual {words(ps[ai + 1])}')
        title = ps[0]
        chk('12', 'title word count <= 15', True, words(title) <= 15, note=f'actual {words(title)}')
        chk('12', 'title free of punctuation', True,
            not any(c in title for c in '.:;,!?'), note=title)
        refs = [t for t in ps if t.strip() and t.strip()[0].isdigit() and '\t' in t]
        nums = [int(t.split('\t')[0]) for t in refs if t.split('\t')[0].isdigit()]
        chk('12', 'reference count <= 60', True, (max(nums) if nums else 0) <= 60,
            note=f'actual {max(nums) if nums else 0} references')
        for sec in ['Data availability', 'Code availability', 'Acknowledgements',
                    'Author contributions', 'Competing interests', 'References',
                    'Figure legends']:
            chk('12', f'section present: {sec}', 'present', 'present' if sec in ps else 'MISSING')
        chk('12', 'Methods comes before Data availability', True,
            ps.index('Methods') < ps.index('Data availability'))
        chk('12', 'References before Figure legends', True,
            ps.index('References') < ps.index('Figure legends'))
        # legend word limits
        fig_starts = [i for i, t in enumerate(ps) if t.startswith('Fig. ') and '|' in t]
        sup_starts = [i for i, t in enumerate(ps) if t.startswith('Supplementary Fig. ') and '|' in t]
        bounds = sorted([(i, 'main') for i in fig_starts] + [(i, 'sup') for i in sup_starts])
        print('         figure legend lengths:')
        for k, (idx, kind) in enumerate(bounds):
            end = bounds[k + 1][0] if k + 1 < len(bounds) else (ps.index('Supplementary information')
                                                                if 'Supplementary information' in ps else len(ps))
            txt = ' '.join(t for t in ps[idx:end] if t.strip())
            print(f'           {ps[idx][:38]:<40} {words(txt):>4} words  ({kind})')
        chk('12', 'all figure legends <= 350 words', True,
            all(words(' '.join(t for t in ps[idx:(bounds[k + 1][0] if k + 1 < len(bounds) else len(ps))]
                              if t.strip())) <= 350 for k, (idx, _) in enumerate(bounds)))

    # ================================================ 13  legend pair order
    # 图注里的两组比较没有主语（写的是 "by subtype"），读者只能靠顺序判断哪个数是
    # BC1。约定必须是 BC1 versus BC2，且与箱线图从左到右的分组顺序一致。
    # 正文有主语（"BC2 showed higher ... (BC2 versus BC1)"），不受此约束，
    # 已在第 8 节按主语顺序核对过。
    print('\n-- 13. legend pair order (BC1 versus BC2) -------------------')
    if not PS:
        print('         [skip] manuscript not available')
    else:
        import re as _re
        NUM = _re.compile(r'([-+]?\d+(?:\.\d+)?)\s+versus\s+([-+]?\d+(?:\.\d+)?)')

        def legend(prefix):
            for t in PS:
                if t.startswith(prefix):
                    return t.replace('\u2212', '-')   # 稿件里的 Unicode 减号
            return ''

        # (图注前缀, 标签, 稿件应写的 BC1 值, BC2 值, 容差)
        pairs = [
            ('B Tumour mutational burden', 'Fig. 2B', 0.93, 1.80, 5e-3),
            ('C Fraction of the genome altered', 'Fig. 2C', 0.217, 0.409, 5e-3),
            ('C Immune-checkpoint score', 'Fig. 3C', 7.36, 8.21, 5e-3),
            ('D Cytolytic activity', 'Fig. 3D', 7.60, 8.30, 5e-3),
            ('E Hypoxia programme score', 'Fig. 3E', -11.4, 10.4, 0.05),
            ('F Stromal compartment score', 'Fig. 3F', 14.59, 14.12, 5e-3),
        ]
        for prefix, tag, v1, v2, tol in pairs:
            t = legend(prefix)
            if not t:
                chk('13', f'{tag} legend paragraph found', 'present', 'MISSING')
                continue
            m = NUM.search(t)
            if not m:
                chk('13', f'{tag} numeric pair found', 'present', 'MISSING')
                continue
            a, b = float(m.group(1)), float(m.group(2))
            ok = chk('13', f'{tag} order', (v1, v2), (a, b), tol=tol)
            if not ok:
                NOTES.append(f'{tag}: legend writes "{a} versus {b}"; the BC1-first '
                             f'convention requires "{v1} versus {v2}"')

    # ====================================================== summary
    print('\n' + '=' * 78)
    print(f'CHECKS RUN : {len(CHECKS)}')
    print(f'FAILURES   : {len(FAILS)}')
    if NOTES:
        print('\nNOTES (not failures, but worth review):')
        for n in NOTES:
            print('  * ' + n)
    if FAILS:
        print('\nFAILED ITEMS:')
        for sec, item, exp, got, note in FAILS:
            print(f'  [{sec}] {item}: manuscript={exp} data={got} {note}')
    print('=' * 78)

    os.makedirs('results/qa', exist_ok=True)
    out = f'results/qa/manuscript_audit_{date.today()}.txt'
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(f'MANUSCRIPT NUMBER AUDIT  ({date.today()})\n')
        fh.write(f'checks={len(CHECKS)}  failures={len(FAILS)}\n\n')
        for sec, item, ok in CHECKS:
            fh.write(f'[{"ok" if ok else "FAIL"}] ({sec}) {item}\n')
        if FAILS:
            fh.write('\nFAILURES\n')
            for sec, item, exp, got, note in FAILS:
                fh.write(f'  [{sec}] {item}: manuscript={exp} data={got} {note}\n')
        if NOTES:
            fh.write('\nNOTES\n')
            for n in NOTES:
                fh.write(f'  * {n}\n')
    print(f'report written to {out}')


if __name__ == '__main__':
    main()
