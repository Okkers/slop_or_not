"""
Dataset_MTT_Detection: reads the CSV produced by data_factory/data_generator.py
(columns: task_prompt, output_text, WAN, label [, generator], plus COOC from data_factory/cooc.py)
and serves one PyG graph per row for the requested split ('train' | 'val' | 'test' | 'pred' = all rows).
--graph_type wan  -> 117-node function-word graphs;  --graph_type cooc -> all-word graphs from COOC.

Graphs are built once and cached in data/cache/ (the cache key includes every graph
setting, so changing e.g. --window or --node_features builds a new cache).
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from data_factory.embeddings import load_vectors, vectors_for
from data_factory.graph_builder import cooc_to_data, text_to_data
from data_factory.splits import get_splits

_MEMORY_CACHE: dict[str, list] = {}   # so train/val/test in one run only build once


def _graph_settings(args) -> dict:
    settings = {
        "graph_type": args.graph_type,
        "window": args.window,
        "node_features": args.node_features,
        "edge_weight": args.edge_weight,
        "undirected": bool(args.undirected),
        "extra_node_cols": args.extra_node_cols,
        "wan_source": args.wan_source,
    }
    if "emb" in args.node_features:   # different word vectors -> different node features
        settings["embeddings"] = args.embeddings
    return settings


def _cache_key(args) -> str:
    st = os.stat(args.data_path)
    with open(args.function_words, "rb") as f:
        fw_hash = hashlib.md5(f.read()).hexdigest()
    payload = json.dumps({"csv": os.path.abspath(args.data_path), "mtime": st.st_mtime, "size": st.st_size,
                          "fw": fw_hash, **_graph_settings(args)}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def build_all_graphs(df: pd.DataFrame, args) -> list:
    feats = [f.strip() for f in args.node_features.split(",") if f.strip()]
    extra_cols = [c.strip() for c in args.extra_node_cols.split(",") if c.strip()] if args.extra_node_cols else []

    wv = None
    if "emb" in feats:
        print(f"[data] loading word vectors '{args.embeddings}' ...")
        wv = load_vectors(args.embeddings)

    if args.graph_type == "wan":
        return _wan_graphs(df, args, feats, extra_cols, wv)
    if args.graph_type == "cooc":
        return _cooc_graphs(df, args, feats, extra_cols, wv)
    raise ValueError(f"unknown graph_type '{args.graph_type}' (wan | cooc)")


def _wan_graphs(df: pd.DataFrame, args, feats, extra_cols, wv) -> list:
    """117-node graphs, rebuilt from output_text with Christina's code (= the WAN column)."""
    use_csv_wan = args.wan_source == "csv"
    if use_csv_wan and "WAN" not in df.columns:
        raise ValueError("--wan_source csv but the CSV has no 'WAN' column")

    emb = None
    if wv is not None:
        with open(args.function_words) as handle:
            fwords = sorted(handle.read().split())     # same order as the WAN nodes
        emb, missing = vectors_for(fwords, wv)          # the same 117 vectors for every essay
        print(f"[data] {emb.shape[1]} numbers per word; function words without a vector: {missing or 'none'}")

    graphs = []
    for i, row in enumerate(df.to_dict("records")):
        extra = [json.loads(row[c]) for c in extra_cols]
        if i and i % 1000 == 0:
            print(f"[data]   {i}/{len(df)}")
        graphs.append(text_to_data(
            str(row["output_text"]), row.get("label"), args.function_words,
            window=args.window,
            precomputed_wan=row["WAN"] if use_csv_wan else None,
            node_features=feats, edge_weight=args.edge_weight, undirected=args.undirected,
            extra=extra, idx=i, emb=emb,
        ))
    return graphs


def _cooc_graphs(df: pd.DataFrame, args, feats, extra_cols, wv) -> list:
    """One graph per COOC cell (made by data_factory/cooc.py); node i = word i of THAT essay."""
    if "COOC" not in df.columns:
        raise ValueError(f"the CSV has no 'COOC' column; add it first with:  "
                         f"uv run python -m data_factory.cooc --data_path {args.data_path}")
    if "onehot" in feats or extra_cols:
        raise ValueError("'onehot' and --extra_node_cols only work with --graph_type wan "
                         "(they need the same 117 words in every graph)")

    graphs, n_words, n_missing = [], 0, 0
    for i, row in enumerate(df.to_dict("records")):
        g = json.loads(row["COOC"])
        if g.get("window") != args.window:
            raise ValueError(f"row {i}: the COOC column was made with window {g.get('window')} but --window is "
                             f"{args.window}; rerun data_factory.cooc with --window {args.window}")
        if i and i % 1000 == 0:
            print(f"[data]   {i}/{len(df)}")
        emb = None
        if wv is not None:
            emb, missing = vectors_for(g["nodes"], wv)   # this essay's own words
            n_words += len(g["nodes"])
            n_missing += len(missing)
        graphs.append(cooc_to_data(
            g, row.get("label"),
            node_features=feats, edge_weight=args.edge_weight, undirected=args.undirected,
            idx=i, emb=emb,
        ))
    if wv is not None:
        print(f"[data] word nodes without a vector (they get zeros): {n_missing} of {n_words} "
              f"({100 * n_missing / max(n_words, 1):.1f}%)")
    return graphs


def load_graphs(df: pd.DataFrame, args) -> list:
    key = _cache_key(args)
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]
    stem = os.path.splitext(os.path.basename(args.data_path))[0]
    path = os.path.join(args.cache_dir, f"{stem}_{key}.pt")
    if args.use_cache and os.path.exists(path):
        graphs = torch.load(path, weights_only=False)
    else:
        print(f"[data] building {len(df)} graphs from {args.data_path} ...")
        graphs = build_all_graphs(df, args)
        if args.use_cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            torch.save(graphs, path)
    _MEMORY_CACHE[key] = graphs
    return graphs


class Dataset_MTT_Detection(Dataset):
    def __init__(self, args, flag: str = "train"):
        assert flag in ("train", "val", "test", "pred")
        self.args = args
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        df = pd.read_csv(self.args.data_path)
        if "label" not in df.columns or "output_text" not in df.columns:
            raise ValueError("CSV needs at least the columns 'output_text' and 'label'")
        df["label"] = df["label"].astype(int)
        self.df = df
        graphs = load_graphs(df, self.args)

        if self.flag == "pred":
            self.indices = list(range(len(df)))
        else:
            self.indices = get_splits(df, self.args)[self.flag]
        self.graphs = [graphs[i] for i in self.indices]

    @property
    def num_node_features(self) -> int:
        return self.graphs[0].num_node_features

    @property
    def labels(self):
        return [int(g.y) for g in self.graphs]

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)