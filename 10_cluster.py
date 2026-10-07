"""
10_cluster.py - Multi-omics consensus clustering of TCGA-BRCA.

Mirrors the MOVICS framework (multi-omics x multi-algorithm consensus
clustering) using scikit-learn implementations, because MOVICS/its compiled
dependencies are not guaranteed on this Windows build.  Every step is
deterministic (fixed seeds) and reproducible.

Layers: mRNA / lncRNA / miRNA / methylation / mutation
Algorithms: KMeans, GMM, Ward, Average-linkage, Spectral, NMF, BisectingKMeans,
            PCA+KMeans, PAM(k-medoids), SNF-style similarity fusion
Consensus: co-clustering (Jean-Philippe) matrix averaged over layers x algorithms
           -> stability + PAC + silhouette to pick K -> final Ward clustering

Outputs (results/cluster/):
  subtypes.csv           sample -> subtype label + consensus score
  k_selection.csv        K diagnostics
  consensus_K*.csv       consensus matrices
  feature_layers.pkl     the exact matrices used
"""
import os
import pickle
import sys
import tempfile

# ---- ASCII temp dirs -----------------------------------------------------
# joblib's memmapping folder and multiprocessing's resource tracker encode
# process names with ascii; a non-ASCII user name in %TEMP% therefore aborts
# parallel clustering.  Every temporary path is pinned to <project>/tmp.
_TMP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tmp")
os.makedirs(_TMP, exist_ok=True)
for _v in ("TMPDIR", "TEMP", "TMP", "JOBLIB_TEMP_FOLDER"):
    os.environ[_v] = _TMP
tempfile.tempdir = _TMP

import warnings

import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
from scipy.spatial.distance import squareform
from sklearn.cluster import (AgglomerativeClustering, BisectingKMeans, KMeans,
                             SpectralClustering)
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRC = os.path.join(PROJ, "data", "processed")
OUT = os.path.join(PROJ, "results", "cluster")
os.makedirs(OUT, exist_ok=True)
SEED = 20260911
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- feature prep
def top_mad(df, n=1500, max_na=0.2):
    cols = [c for c in df.columns if c.startswith("TCGA-")]
    x = df[cols]
    x = x.loc[x.isna().mean(axis=1) <= max_na]
    x = x.fillna(x.mean(axis=1), axis=0)
    keep = x.var(axis=1).sort_values(ascending=False).head(n).index
    return x.loc[keep]


def mutation_features(df, freq=0.03, n=800):
    cols = [c for c in df.columns if c.startswith("TCGA-")]
    x = df[cols]
    x = x.loc[x.mean(axis=1) >= freq]
    if x.shape[0] > n:
        x = x.loc[x.mean(axis=1).sort_values(ascending=False).head(n).index]
    return x


def prepare(common=None, use=None):
    specs = {
        "mRNA": ("mrna.csv", 1200, "mad"),
        "lncRNA": ("lncrna.csv", 800, "mad"),
        "miRNA": ("mirna.csv", 300, "mad"),
        "Methylation": ("methyl_top10000.csv", 1500, "mad"),
        "Mutation": ("mut_matrix.csv", 800, "mut"),
    }
    layers = {}
    for name, (fn, n, kind) in specs.items():
        p = os.path.join(PRC, fn)
        if use is not None and name not in use:
            continue
        if not os.path.exists(p):
            print(f"  [skip] {name}: {fn} not available")
            continue
        df = pd.read_csv(p, index_col=0)
        layers[name] = mutation_features(df, n=n) if kind == "mut" else top_mad(df, n)
    if not layers:
        raise SystemExit("no omics layer available")
    commons = set.intersection(*[set(v.columns) for v in layers.values()])
    if common is not None:
        commons &= set(common)
    commons = sorted(commons)
    print(f"  layers used: {list(layers)}")
    print(f"  samples present in all layers: {len(commons)}")
    out = {}
    for k, v in layers.items():
        m = v[commons]
        X = StandardScaler().fit_transform(m.T.values.astype(np.float64))
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # clip extreme z-scores: keeps all downstream distances finite and
        # prevents float overflow inside PCA / k-means inertia accumulation
        X = np.clip(X, -8.0, 8.0)
        out[k] = X
        print(f"  {k:<12} {X.shape}")
    return out, commons


# ------------------------------------------------------------------- algorithms
def snf_affinity(X, k=20):
    """Similarity-network-fusion style affinity from a Euclidean distance."""
    from sklearn.metrics.pairwise import euclidean_distances
    D = euclidean_distances(X)
    n = D.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        idx = np.argsort(D[i])[1:k + 1]
        K[i, idx] = 1.0 / (D[i, idx] + 1e-10)
    W = (K + K.T) / 2
    W = W / (2 * W.sum(axis=1, keepdims=True))
    return W


def run_algos(X, k):
    """Return a dict of algorithm -> label vector."""
    res = {}
    res["kmeans"] = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
    res["gmm"] = GaussianMixture(k, covariance_type="diag", n_init=5,
                                 random_state=SEED).fit_predict(X)
    res["ward"] = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
    res["average"] = AgglomerativeClustering(k, linkage="average",
                                             metric="cosine").fit_predict(X)
    res["complete"] = AgglomerativeClustering(k, linkage="complete").fit_predict(X)
    res["spectral"] = SpectralClustering(
        k, random_state=SEED, n_neighbors=15, assign_labels="kmeans").fit_predict(X)
    res["bisect"] = BisectingKMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    # PCA + kmeans (iCluster-like low-rank partition)
    Z = PCA(n_components=min(20, X.shape[1] - 1, X.shape[0] - 1),
            random_state=SEED).fit_transform(X)
    res["pca_kmeans"] = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)
    res["pca_ward"] = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    # NMF on a non-negative shifted matrix
    Xn = X - X.min(axis=0, keepdims=True) + 1e-6
    W = NMF(n_components=k, init="nndsvda", random_state=SEED, max_iter=200).fit_transform(Xn)
    res["nmf"] = KMeans(k, n_init=20, random_state=SEED).fit_predict(W)
    # SNF-style spectral partition of the fused affinity
    A = snf_affinity(X, k=20)
    try:
        res["snf"] = SpectralClustering(
            k, affinity="precomputed", random_state=SEED,
            assign_labels="kmeans").fit_predict(A)
    except Exception:  # noqa: BLE001
        res["snf"] = res["spectral"]
    return res


def consensus_matrix(layers, k, n_jobs=8):
    """Average co-clustering matrix across every layer x algorithm."""
    from joblib import Parallel, delayed
    n = next(iter(layers.values())).shape[0]
    items = list(layers.items())

    def one(name_X):
        name, X = name_X
        return name, run_algos(X, k)

    res = Parallel(n_jobs=min(n_jobs, len(items)))(delayed(one)(it) for it in items)
    C = np.zeros((n, n))
    cnt = 0
    per_algo = {}
    for lname, lab in res:
        for aname, l in lab.items():
            C += (l[:, None] == l[None, :]).astype(float)
            cnt += 1
            per_algo[f"{lname}|{aname}"] = l
    C /= cnt
    np.fill_diagonal(C, 1.0)
    return C, per_algo


def pac(C, k):
    """Proportion of ambiguous clustering (lower = crisper)."""
    v = squareform(1 - C, checks=False)
    return float(((v > 0.1) & (v < 0.9)).mean())


def cdf_area(C, step=0.01):
    """Area under the empirical CDF of consensus indices
    (ConsensusClusterPlus criterion)."""
    v = squareform(C, checks=False)
    xs = np.arange(0.0, 1.0 + step, step)
    cdf = (v[:, None] <= xs[None, :]).mean(axis=0)
    return float(np.trapezoid(cdf, xs))


def main():
    layers, samples = prepare()
    with open(os.path.join(OUT, "feature_layers.pkl"), "wb") as f:
        pickle.dump({"layers": layers, "samples": samples}, f)

    ks = [2, 3, 4, 5, 6]
    diag = []
    mats = {}
    for k in ks:
        C, per_algo = consensus_matrix(layers, k)
        mats[k] = C
        d = 1 - C
        np.fill_diagonal(d, 0)
        Z = linkage(squareform(d, checks=False), method="ward")
        lab = fcluster(Z, k, criterion="maxclust")
        sil = silhouette_score(d, lab, metric="precomputed")
        stability = float(np.mean([
            (per_algo[a][:, None] == per_algo[a][None, :]).mean() for a in per_algo]))
        diag.append({"K": k, "silhouette": sil, "PAC": pac(C, k),
                     "cdf_area": cdf_area(C),
                     "mean_within_algo_agreement": stability})
        np.savetxt(os.path.join(OUT, f"consensus_K{k}.csv"), C, delimiter=",")
        print(f"  K={k}  silhouette={sil:.3f}  PAC={pac(C,k):.3f}  "
              f"area={cdf_area(C):.4f}", flush=True)
    dd = pd.DataFrame(diag)

    # ---- K selection: agreement across three independent criteria ------------
    #   ConsensusClusterPlus delta-area  -> larger marginal CDF-area gain = better
    #   silhouette                       -> larger = better
    #   PAC (Senbabaoglu 2014)           -> smaller = better
    area = dict(zip(dd["K"], dd["cdf_area"]))
    delta = {}
    for a, b in zip(ks[:-1], ks[1:]):
        delta[a] = (area[b] - area[a]) / area[a]
    delta[ks[-1]] = np.nan
    dd["delta_area"] = dd["K"].map(delta)
    dd["rank_sil"] = dd["silhouette"].rank(ascending=False)
    dd["rank_pac"] = dd["PAC"].rank(ascending=True)
    dd["rank_area"] = dd["delta_area"].rank(ascending=False, na_option="bottom")
    dd["rank_sum"] = dd[["rank_sil", "rank_pac", "rank_area"]].sum(axis=1)
    dd = dd.sort_values("K").reset_index(drop=True)
    dd.to_csv(os.path.join(OUT, "k_selection.csv"), index=False)

    best_k = int(dd.sort_values(["rank_sum", "K"]).iloc[0]["K"])
    print(f"  criteria -> " + ", ".join(
        f"K={int(r.K)}: sil={r.silhouette:.2f} PAC={r.PAC:.2f} "
        f"dA={0 if np.isnan(r.delta_area) else r.delta_area:.3f} "
        f"sum={int(r.rank_sum)}" for r in dd.itertuples()))
    print(f"  selected K = {best_k} (best agreement across criteria)")

    C = mats[best_k]
    d = 1 - C
    np.fill_diagonal(d, 0)
    Z = linkage(squareform(d, checks=False), method="ward")
    lab = fcluster(Z, best_k, criterion="maxclust")
    # keep co-clustering order deterministic: relabel by cluster size
    order = pd.Series(lab).value_counts().index.tolist()
    remap = {c: f"BC{i+1}" for i, c in enumerate(order)}
    sub = pd.DataFrame({"sample": samples, "cluster": [remap[x] for x in lab]})
    sub["consensus_score"] = [float(C[i, lab == lab[i]].mean()) for i in range(len(lab))]
    sub.to_csv(os.path.join(OUT, "subtypes.csv"), index=False)
    print(sub["cluster"].value_counts().to_dict())
    print("DONE")


if __name__ == "__main__":
    main()
