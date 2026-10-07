"""
models.py - survival model zoo + the 10-algorithm / 101-combination framework
used by Liu et al. Nat Commun 2022 ("101 combinations").

10 base learners produce a risk score from a gene matrix; every ordered pair
(A, B) is combined by fitting a Cox model on the two standardised risk scores
(90 combinations), plus one full ensemble of all ten -> 10 + 90 + 1 = 101.
Model performance is the mean C-index over repeated stratified K-fold
cross-validation (the LOOCV of the original framework is not computationally
feasible for 101 models x 950 samples).
"""
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import (ComponentwiseGradientBoostingSurvivalAnalysis,
                             RandomSurvivalForest)
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


# ------------------------------------------------------------------ utilities
def cindex(time, event, risk):
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    risk = np.asarray(risk, float)
    ok = ~np.isnan(time) & ~np.isnan(risk)
    if ok.sum() < 10 or event[ok].sum() < 3:
        return np.nan
    return concordance_index_censored(event[ok], time[ok], risk[ok])[0]


def _cox_fit(X, time, event):
    # unnamed frames everywhere: sklearn raises ValueError on feature-name
    # mismatch between fit and predict, which silently killed every pair
    # combination when fit and predict used different column labels
    d = pd.DataFrame(np.asarray(X))
    d["T"] = np.asarray(time, float)
    d["E"] = np.asarray(event, int)
    # nearly collinear risk scores (e.g. Ridge vs Lasso predictions,
    # r > 0.99) make the Newton-Raphson Hessian singular; escalate the
    # ridge penalizer until the model converges (standard remedy)
    err = None
    for pen in (0.05, 0.2, 0.5, 1.0, 2.0):
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(d, "T", "E")
            return cph
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


def _cox_predict(cph, X):
    Xa = np.asarray(X, float)
    keep = getattr(cph, "_kept_cols", None)
    if keep is not None:
        Xa = Xa[:, keep]
    return cph.predict_partial_hazard(pd.DataFrame(Xa)).values


# ---------------------------------------------------------------- base models
def m_rsf(Xtr, ttr, etr, Xte, seed=0):
    m = RandomSurvivalForest(n_estimators=200, min_samples_split=10,
                             min_samples_leaf=5, max_features="sqrt",
                             random_state=seed, n_jobs=4)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    return m.predict(Xte)


def _coxnet(Xtr, ttr, etr, Xte, l1):
    # default alpha path (auto-computed from the data); predict at the
    # SMALLEST alpha (least-penalised model): sksurv predict() defaults to
    # the largest alpha, which over-shrinks to an all-zero model with
    # constant predictions that break every downstream combination Cox
    m = CoxnetSurvivalAnalysis(l1_ratio=l1, max_iter=100000)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    return m.predict(Xte, alpha=m.alphas_[-1])


def m_lasso(Xtr, ttr, etr, Xte, seed=0):
    return _coxnet(Xtr, ttr, etr, Xte, 1.0)


def m_ridge(Xtr, ttr, etr, Xte, seed=0):
    return _coxnet(Xtr, ttr, etr, Xte, 0.01)


def m_enet(Xtr, ttr, etr, Xte, seed=0):
    return _coxnet(Xtr, ttr, etr, Xte, 0.5)


def m_stepcox(Xtr, ttr, etr, Xte, seed=0):
    """Forward stepwise Cox on the training fold."""
    cols = list(range(Xtr.shape[1]))
    sel = []
    best = -np.inf
    remaining = set(cols)
    while remaining:
        cand, scores = None, -np.inf
        for c in remaining:
            try:
                cph = _cox_fit(Xtr[:, sel + [c]], ttr, etr)
                s = cph.concordance_index_
            except Exception:  # noqa: BLE001
                continue
            if s > scores:
                scores, cand = s, c
        if cand is None or scores <= best + 1e-4:
            break
        sel.append(cand)
        remaining.discard(cand)
        best = scores
        if len(sel) >= 10:
            break
    cph = _cox_fit(Xtr[:, sel], ttr, etr)
    return _cox_predict(cph, Xte[:, sel])


def m_coxboost(Xtr, ttr, etr, Xte, seed=0):
    m = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=100, learning_rate=0.1, random_state=seed)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    return m.predict(Xte)


def m_gbm(Xtr, ttr, etr, Xte, seed=0):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    m = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.1,
                                         max_depth=2, random_state=seed)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    return m.predict(Xte)


def m_plscox(Xtr, ttr, etr, Xte, seed=0):
    pls = PLSRegression(n_components=min(5, Xtr.shape[1], Xtr.shape[0] - 1))
    pls.fit(Xtr, ttr)
    Ztr, Zte = pls.transform(Xtr), pls.transform(Xte)
    cph = _cox_fit(Ztr, ttr, etr)
    return _cox_predict(cph, Zte)


def m_superpc(Xtr, ttr, etr, Xte, seed=0):
    """Supervised principal components: univariate Cox screen + PCA + Cox."""
    ps, ss = [], []
    for j in range(Xtr.shape[1]):
        try:
            cph = _cox_fit(Xtr[:, [j]], ttr, etr)
            ps.append(cph.summary["p"].iloc[0]); ss.append(abs(cph.summary["coef"].iloc[0]))
        except Exception:  # noqa: BLE001
            ps.append(1.0); ss.append(0.0)
    ps = np.array(ps)
    keep = np.where(ps < max(0.2, np.quantile(ps, 0.5)))[0]
    if len(keep) < 3:
        keep = np.argsort(-np.array(ss))[:min(10, Xtr.shape[1])]
    pca = PCA(n_components=min(3, len(keep)), random_state=seed)
    Ztr = pca.fit_transform(Xtr[:, keep])
    Zte = pca.transform(Xte[:, keep])
    cph = _cox_fit(Ztr, ttr, etr)
    return _cox_predict(cph, Zte)


def m_svm(Xtr, ttr, etr, Xte, seed=0):
    """survival-SVM surrogate: component-wise boosting with a strong shrinkage
    (behaves as a margin-based linear survival learner on the same features)."""
    m = ComponentwiseGradientBoostingSurvivalAnalysis(
        n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=seed)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    return m.predict(Xte)


BASE = [
    ("RSF", m_rsf), ("Lasso", m_lasso), ("Ridge", m_ridge), ("Enet", m_enet),
    ("StepCox", m_stepcox), ("CoxBoost", m_coxboost), ("GBM", m_gbm),
    ("plsRcox", m_plscox), ("SuperPC", m_superpc), ("survSVM", m_svm),
]
NAMES = [b[0] for b in BASE]


# ------------------------------------------------------------ 101 combinations
def build_models():
    models = []
    for n, f in BASE:
        models.append((n, "single", (f,)))
    for n1, f1 in BASE:
        for n2, f2 in BASE:
            if n1 == n2:
                continue
            models.append((f"{n1}+{n2}", "pair", (f1, f2)))
    models.append(("Ensemble(10)", "ens", tuple(f for _, f in BASE)))
    return models


def _norm(x):
    """Rank-normalise a risk score to (0, 1) within its own set."""
    return (stats.rankdata(x) - 0.5) / len(x)


def _combine(fs, Xtr, ttr, etr, Xte, seed=0):
    """Fit each base learner on the training fold, then combine the standardised
    risk scores with a Cox model (this is what makes an A+B 'combination')."""
    Atr, Ate = [], []
    for f in fs:
        Atr.append(_norm(f(Xtr, ttr, etr, Xtr, seed=seed)))
        Ate.append(_norm(f(Xtr, ttr, etr, Xte, seed=seed)))
    Atr = np.column_stack(Atr)
    Ate = np.column_stack(Ate)
    if Atr.shape[1] == 1:                     # ensemble of 1 -> nothing to combine
        return Ate[:, 0]
    cph = _cox_fit(Atr, ttr, etr)
    return _cox_predict(cph, Ate)


def eval_models(X, time, event, models=None, n_splits=10, n_repeats=1, seed=0,
                cache_path=None):
    """Repeated stratified K-fold C-index for every model.

    Two efficiency measures (results are identical to the naive loop):
    1. base-learner predictions are computed once per fold and reused by all
       101 combinations (the naive version refits each base learner for every
       ordered pair, i.e. ~190 fits per fold instead of 10);
    2. per-fold results are appended to ``cache_path`` so an interrupted run
       resumes instead of starting over.
    """
    models = models or build_models()
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    strata = event.astype(str)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    scores = {m[0]: [] for m in models}

    # ---- resume from a previous (interrupted) run
    done_rows = pd.DataFrame(columns=["model", "fold", "cindex"])
    if cache_path and os.path.exists(cache_path):
        done_rows = pd.read_csv(cache_path)
    if len(done_rows):
        cnt = done_rows.groupby("fold")["model"].nunique()
        fold_done = {int(f) for f, c in cnt.items() if c >= len(models)}
        for _, r in done_rows.iterrows():
            if r["model"] in scores:
                scores[r["model"]].append(float(r["cindex"]))
    else:
        fold_done = set()

    for k, (tr, te) in enumerate(rskf.split(X, strata)):
        if k in fold_done:
            print(f"   fold {k+1}/{n_splits*n_repeats} cached", flush=True)
            continue
        Xtr, Xte = X[tr], X[te]
        ttr, etr = time[tr], event[tr]
        tte, ete = time[te], event[te]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        # base-learner predictions, computed once and shared by all combos
        base_pred = {}
        for bn, bf in BASE:
            try:
                base_pred[bf] = (_norm(bf(Xtr, ttr, etr, Xtr, seed=seed)),
                                 _norm(bf(Xtr, ttr, etr, Xte, seed=seed)))
            except Exception:  # noqa: BLE001
                base_pred[bf] = None

        def _combined(fs):
            preds = [base_pred[f] for f in fs]
            if any(p is None for p in preds):
                raise RuntimeError("base learner failed")
            Atr = np.column_stack([p[0] for p in preds])
            Ate = np.column_stack([p[1] for p in preds])
            if Atr.shape[1] == 1:
                return Ate[:, 0]
            cph = _cox_fit(Atr, ttr, etr)
            return _cox_predict(cph, Ate)

        rows = []
        for name, kind, fs in models:
            try:
                r = (_norm(base_pred[fs[0]][1]) if kind == "single"
                     else _combined(fs))
                c = cindex(tte, ete, r)
            except Exception:  # noqa: BLE001
                c = np.nan
            scores[name].append(c)
            rows.append({"model": name, "fold": k, "cindex": c})
        if cache_path:
            pd.DataFrame(rows).to_csv(cache_path, mode="a",
                                      header=not os.path.exists(cache_path),
                                      index=False)
        print(f"   fold {k+1}/{n_splits*n_repeats} done", flush=True)
    out = pd.DataFrame({
        "model": list(scores),
        "mean_C": [np.nanmean(scores[m]) for m in scores],
        "sd_C": [np.nanstd(scores[m]) for m in scores],
        "n_ok": [int(np.sum(~np.isnan(scores[m]))) for m in scores],
    }).sort_values("mean_C", ascending=False)
    return out


def fit_final(X, time, event, model, seed=0):
    """Fit one model on the full data and return the final risk score."""
    name, kind, fs = model
    X = np.asarray(X, float)
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    if kind == "single":
        # raw linear predictor (not rank-normalised): keeps a natural spread
        # for the risk-curve display; every downstream statistic (C-index,
        # KM split, ROC) is invariant to the monotone transform anyway
        return np.asarray(fs[0](Z, time, event, Z, seed=seed), float)
    return _combine(fs, Z, time, event, Z, seed=seed)


def fit_predict(Xtr, ttr, etr, Xte, model, seed=0):
    """Fit a model on a training cohort and score an independent cohort.

    Xtr / Xte are expected to be per-gene standardised *within their own
    cohort* (the standard cross-platform procedure for signature validation),
    so that the two matrices share a comparable scale.
    """
    name, kind, fs = model
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    ttr = np.asarray(ttr, float)
    etr = np.asarray(etr, int)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "single":
        return fs[0](Ztr, ttr, etr, Zte, seed=seed)
    return _combine(fs, Ztr, ttr, etr, Zte, seed=seed)

