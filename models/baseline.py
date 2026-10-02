"""
Non-GNN baselines required by the course (see run_baselines.py for the driver):

  SVM         support vector machine on graph-level feature vectors extracted from each WAN
              (--svm_features, any '+'-combination of):
                  freq        function-word frequencies (per 100 tokens), the classic stylometry vector
                  wan         the flattened WAN matrix (n*n transition weights)
                  graphstats  hand-crafted graph-level statistics of the WAN (size, density,
                              reciprocity, transition entropy, spectral gap, ... see graph_stats)

  WL          Weisfeiler-Lehman subtree graph kernel + SVM (Shervashidze et al., JMLR 2011):
              the "traditional graph machine learning" method. Node labels = function word;
              each iteration relabels a node by (own label, sorted out-neighbour labels,
              sorted in-neighbour labels). A graph = histogram of all labels seen. We train a
              linear SVM on the L2-normalised histograms, which is exactly an SVM with the
              (cosine-normalised) WL kernel but scales to large datasets.

  RelEnt      The original WAN classifier of Segarra, Eisen & Ribeiro (2015): build one
              "profile" WAN per class from all training texts, then assign each test text
              to the class whose profile has the smallest relative entropy
                  H(P, Q) = sum_ij pi_i * P_ij * log(P_ij / Q_ij)
              (pi = stationary distribution of the text's WAN; cells where P_ij or Q_ij is 0
              are skipped). No training beyond averaging, so it shows how far plain WAN
              similarity gets.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
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


def make_svm(kernel: str, C: float):
    if kernel == "linear":
        return make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=20000))
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma="scale"))


# ----------------------------------------------------------------------------- WL kernel
def wl_features(W_list, counts_list, iterations: int = 2, threshold: float = 0.1):
    """Sparse WL label-histogram matrix (graphs x labels)."""
    compress: dict = {}
    rows, cols, vals = [], [], []
    for g, (W, counts) in enumerate(zip(W_list, counts_list)):
        A = W >= max(threshold, 1e-12)
        nodes = np.flatnonzero((counts > 0) | A.any(0) | A.any(1))
        out_nb = {i: np.flatnonzero(A[i]) for i in nodes}
        in_nb = {i: np.flatnonzero(A[:, i]) for i in nodes}
        labels = {i: ("w", int(i)) for i in nodes}                 # iteration 0: the word itself
        hist: dict = {}
        for it in range(iterations + 1):
            for i in nodes:
                key = compress.setdefault((it, labels[i]), len(compress))
                hist[key] = hist.get(key, 0) + 1
            if it == iterations:
                break
            labels = {i: (labels[i],
                          tuple(sorted(compress[(it, labels[j])] for j in out_nb[i])),
                          tuple(sorted(compress[(it, labels[j])] for j in in_nb[i]))) for i in nodes}
        for k, v in hist.items():
            rows.append(g); cols.append(k); vals.append(v)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(W_list), len(compress)), dtype=float)


def make_wl_svm(C: float):
    return make_pipeline(Normalizer(norm="l2"), LinearSVC(C=C, max_iter=20000))


# ----------------------------------------------------------------------------- RelEnt
def stationary(P: np.ndarray, iters: int = 200) -> np.ndarray:
    n = P.shape[0]
    Q = P.copy()
    sinks = Q.sum(1) == 0
    Q[sinks] = 1.0 / n                                            # "eliminate sinks" as in the WAN script
    pi = np.full(n, 1.0 / n)
    for _ in range(iters):
        pi = pi @ Q
    return pi / pi.sum()


def relative_entropy(P: np.ndarray, Q: np.ndarray, pi: np.ndarray) -> float:
    m = (P > 0) & (Q > 0)
    i, j = np.nonzero(m)
    return float(np.sum(pi[i] * P[i, j] * np.log(P[i, j] / Q[i, j])))


class RelEntClassifier:
    """Segarra et al. (2015) nearest-profile classifier. Profiles come from summed raw scores."""

    def fit(self, raw_list, y):
        y = np.asarray(y)
        self.profiles = []
        for c in (0, 1):
            S = np.sum([raw_list[k] for k in np.flatnonzero(y == c)], axis=0)
            rs = S.sum(1, keepdims=True)
            self.profiles.append(np.divide(S, rs, out=np.zeros_like(S), where=rs != 0))
        return self

    def decision_function(self, W_list):
        """H(text, human profile) - H(text, machine profile): > 0 means closer to machine."""
        out = []
        for P in W_list:
            pi = stationary(P)
            out.append(relative_entropy(P, self.profiles[0], pi) - relative_entropy(P, self.profiles[1], pi))
        return np.array(out)
