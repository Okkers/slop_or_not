"""
Turns one text into a PyTorch-Geometric `Data` graph, in one of two ways (--graph_type):

wan   text_to_data: every graph has the SAME node set: node i == function word i (alphabetical
      order from function_words.txt). Words that do not occur in the text stay in the graph as
      isolated nodes with zero count. This keeps graphs aligned across excerpts, which is
      what lets GNNExplainer masks be averaged into class-level "fingerprints" (RQ2).
cooc  cooc_to_data: one node per unique word of THIS text (read from the COOC column made by
      data_factory/cooc.py), so every graph has its own words and its own size.
Both end in wan_to_data, so node features and edges are made the same way for both.

Node features (choose with --node_features, comma separated, order is kept):
    count   log(1 + raw count of the function word in the text)
    freq    occurrences per 100 tokens (normalised frequency)
    emb     pre-trained word vector of the word (default: GloVe, 100 numbers; see
            data_factory/embeddings.py and --embeddings); unknown words get zeros
    onehot  identity of the function word (117 numbers); the old setting, kept for comparison (wan only)
    + any --extra_node_cols: CSV columns holding a JSON list with one value per
      function word (e.g. a future per-word perplexity column) (wan only)

Edges (--edge_weight), the same for both graph types:
    wan     directed edge i->j for every W[i, j] > 0, weight = W[i, j]
    binary  same edges, weight = 1  (the "unweighted" ablation)
--undirected symmetrises the WAN first: W <- (W + W^T) / 2
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch_geometric.data import Data

from vis import utils as wan_utils   # Christina's WAN code (gabrielegan.com script)

wan_utils.VERBOSE = False            # don't print every comparison while building thousands of graphs

NODE_FEATURES = ("count", "freq", "emb", "onehot")


def node_feature_matrix(counts: np.ndarray, n_tokens: int, features: list[str],
                        extra: list[np.ndarray] | None = None, emb: np.ndarray | None = None) -> np.ndarray:
    n = len(counts)
    cols = []
    for f in features:
        if f == "count":
            cols.append(np.log1p(counts)[:, None])
        elif f == "freq":
            cols.append((100.0 * counts / max(n_tokens, 1))[:, None])
        elif f == "emb":
            if emb is None:
                raise ValueError("node feature 'emb' needs word vectors (loaded in data_loader.build_all_graphs)")
            cols.append(emb)
        elif f == "onehot":
            cols.append(np.eye(n))
        else:
            raise ValueError(f"unknown node feature '{f}', choose from {NODE_FEATURES}")
    for e in extra or []:
        e = np.asarray(e, dtype=np.float64).reshape(n, -1)
        cols.append(e)
    if not cols:  # no features at all -> constant feature, the GNN only sees structure
        cols.append(np.ones((n, 1)))
    return np.concatenate(cols, axis=1)


def wan_to_data(W: np.ndarray, counts: np.ndarray, n_tokens: int, label: int | None, *,
                node_features: list[str], edge_weight: str = "wan", undirected: bool = False,
                extra: list[np.ndarray] | None = None, idx: int | None = None,
                emb: np.ndarray | None = None) -> Data:
    W = np.asarray(W, dtype=np.float64)
    if undirected:
        W = 0.5 * (W + W.T)
    src, dst = np.nonzero(W)
    if edge_weight == "wan":
        w = W[src, dst]
    elif edge_weight == "binary":
        w = np.ones(len(src))
    else:
        raise ValueError(f"unknown edge_weight '{edge_weight}' (wan|binary)")

    x = node_feature_matrix(counts, n_tokens, node_features, extra, emb)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_weight=torch.tensor(w, dtype=torch.float32),
    )
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)
    if idx is not None:
        data.idx = torch.tensor([int(idx)], dtype=torch.long)   # row in the CSV, to map predictions back
    return data


def text_to_data(text: str, label: int | None, function_words_file: str, *, window: int = 5,
                 precomputed_wan: str | None = None, **kwargs) -> Data:
    """Build the graph for one excerpt. If `precomputed_wan` (the JSON string from the
    CSV's WAN column) is given it is used as the adjacency; counts always come from text."""
    W, counts, n_tok = wan_utils.construct_wan_from_text(
        text, windowWidth=window, functionWordsFile=function_words_file, return_counts=True)
    W, counts = np.asarray(W, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    if precomputed_wan is not None:
        W_csv = np.asarray(json.loads(precomputed_wan), dtype=np.float64)
        if W_csv.shape != W.shape:
            raise ValueError(f"WAN column has shape {W_csv.shape} but function-word list gives {W.shape}; "
                             "use the same function_words.txt that generated the CSV")
        W = W_csv
    return wan_to_data(W, counts, n_tok, label, **kwargs)


def cooc_to_data(g: dict, label: int | None, **kwargs) -> Data:
    """Build the graph for one COOC cell (the parsed JSON made by data_factory/cooc.py).
    Node i = word g["nodes"][i]. The edge list is unpacked into a square matrix, so the rest
    is exactly the WAN path (wan_to_data)."""
    n = len(g["nodes"])
    W = np.zeros((n, n))
    for a, b, w in g["edges"]:
        W[a, b] = w
    return wan_to_data(W, np.asarray(g["counts"], dtype=np.float64), g["n_tokens"], label, **kwargs)