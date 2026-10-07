"""
SVM baseline required by the course (see run_baselines.py for the driver).

Support vector machine on a feature vector extracted from each essay's WAN
(--svm_features, any '+'-combination of):
    freq        function-word frequencies (per 100 tokens), the classic stylometry vector
    wan         the flattened WAN matrix (n*n transition weights)
    graphstats  hand-crafted graph-level statistics of the WAN (size, density,
                reciprocity, transition entropy, spectral gap, ... see graph_stats)
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


# ----------------------------------------------------------------------------- features
def graph_stats(W: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Graph-level statistics of one (row-normalised, directed, weighted) WAN."""
    n = W.shape[0]
    A = (W > 0).astype(float)
    active = counts > 0
    n_active = active.sum()
    n_edges = A.sum()
    density = n_edges / max(n_active * n_active, 1)
    recip = (A * A.T).sum() / max(n_edges, 1)                      # fraction of edges with a reverse edge
    self_loop_mass = np.trace(W) / max((W.sum(1) > 0).sum(), 1)
    out_deg, in_deg = A.sum(1), A.sum(0)
    in_strength = W.sum(0)
    rows = W[W.sum(1) > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        row_ent = -np.nansum(np.where(rows > 0, rows * np.log(rows), 0), axis=1) if len(rows) else np.zeros(1)
    asym = np.linalg.norm(W - W.T) / max(np.linalg.norm(W), 1e-12)
    U = np.maximum(A, A.T)
    np.fill_diagonal(U, 0)
    deg_u = U.sum(1)
    triangles = np.trace(U @ U @ U) / 6
    triples = (deg_u * (deg_u - 1)).sum() / 2
    transitivity = 3 * triangles / triples if triples > 0 else 0.0
    eig = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    slem = eig[1] if len(eig) > 1 else 0.0                         # second largest eigenvalue modulus
    return np.array([
        n_active, n_edges, density, recip, self_loop_mass,
        out_deg.mean(), out_deg.std(), in_deg.std(), in_deg.max(),
        in_strength.std(), in_strength.max(),
        row_ent.mean(), row_ent.std(), asym, transitivity, slem,
    ], dtype=float)


def feature_matrix(W_list, counts_list, ntok_list, kinds: str) -> np.ndarray:
    blocks = []
    for kind in kinds.split("+"):
        kind = kind.strip()
        if kind == "freq":
            blocks.append(np.stack([100 * c / max(t, 1) for c, t in zip(counts_list, ntok_list)]))
        elif kind == "wan":
            blocks.append(np.stack([W.ravel() for W in W_list]))
        elif kind == "graphstats":
            blocks.append(np.stack([graph_stats(W, c) for W, c in zip(W_list, counts_list)]))
        else:
            raise ValueError(f"unknown SVM feature '{kind}' (freq | wan | graphstats)")
    return np.concatenate(blocks, axis=1)


# ----------------------------------------------------------------------------- models
def make_svm(kernel: str, C: float):
    if kernel == "linear":
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=20000))
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma="scale"))


def make_rf(n_estimators: int = 500,
            max_depth: int | None = None,
            min_samples_leaf: int = 1,
            max_features: str | float | None = "sqrt",
            class_weight: str | None = "balanced",
            seed: int = 0,
            n_jobs: int = -1):
    """Random Forest on the same feature vectors as the SVM"""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=n_jobs,
    )