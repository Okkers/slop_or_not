import numpy as np
import math
import torch
from itertools import combinations
from collections import defaultdict
import pandas as pd
from scipy.stats import fisher_exact
import ast
import os
import torch


def get_node_importance(node_mask):
    """
    Convert node_mask into node importance if needed. 
    """
    if node_mask is None:
        return None

    mask = torch.as_tensor(node_mask).detach().cpu()

    if mask.ndim == 2:
        return mask.mean(dim=1)

    if mask.ndim == 1:
        return mask

    raise ValueError(f"Unexpected node mask shape {mask.shape}")

def select_top_nodes(result, fraction=0.10, top_k=None):
    """
    Same as select_top_edges, but now for nodes.
    """ 
    node_mask = result["node_mask"]
    if node_mask is None:
        raise ValueError("Explanation has no node mask")

    node_mask = torch.as_tensor(node_mask)

    importance = get_node_importance(node_mask)

    N = len(importance)

    if top_k is not None:
        k = min(top_k, N)
    else:
        k = math.ceil(fraction * N)

    top_indices = torch.argsort(
        importance,
        descending=True
    )[:k]

    return {
        "row": result["row"],
        "label": result["label"],
        "pred": result["pred"],
        "selected_nodes": set(top_indices.tolist()),
        "node_importance": importance
    }

def aggregate_important_words(results, selected_results, fwords):
    graphs = []
    all_words = set()

    for original, selected in zip(results, selected_results):
        if original["label"] != original["pred"]:
            continue

        N = original["x"].shape[0]

        original_nodes = set(range(N))

        selected_nodes = selected["selected_nodes"]

        all_words.update(selected_nodes)

        graphs.append({
            "label": original["label"],
            "original_nodes": original_nodes,
            "selected_nodes": selected_nodes
        })

    rows = []
    for node in sorted(all_words):
        row = {
            "node": node,
            "word": fwords[node]
        }

        for cls, name in [(0, "human"), (1, "machine")]:
            group = [
                g for g in graphs
                if g["label"] == cls
            ]

            present = sum(
                node in g["original_nodes"]
                for g in group
            )

            selected = sum(
                node in g["selected_nodes"]
                for g in group
            )

            row[f"{name}_present"] = present
            row[f"{name}_selected"] = selected

            row[f"{name}_rate"] = (
                selected / present
                if present > 0
                else float("nan")
            )

        row["difference"] = (row["machine_rate"] - row["human_rate"])

        rows.append(row)

    df = pd.DataFrame(rows)

    return df

def analyze_important_nodes(results, folder, path_base, fwords, fraction=0.10, top_k=None):
    os.makedirs(folder, exist_ok=True)

    fraction_name = (
        f"k{top_k}nd" if top_k is not None
        else f"{fraction:.2f}nd".replace(".","")
    )

    selected_results = [
        select_top_nodes(
            result,
            fraction=fraction,
            top_k=top_k
        )
        for result in results
    ]

    node_df = aggregate_important_words(results, selected_results, fwords)

    selected_path = os.path.join(
        folder,
        f"{path_base}_selnod{fraction_name}.pth"
    )

    node_df_path = os.path.join(
        folder,
        f"{path_base}_nc{fraction_name}.csv"
    )

    torch.save(selected_results, selected_path)
    node_df.to_csv(node_df_path, index=False)

    print(f"Unique important words: {len(node_df)}")

    return node_df

def get_feature_importance(node_mask):
    """
    Aggregate the node-feature mask into one importance
    score per feature.
    """
    node_mask = torch.as_tensor(node_mask).detach().cpu()

    if node_mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D node mask, got {tuple(node_mask.shape)}"
        )

    return node_mask.mean(dim=0)


def select_top_features(result, fraction=0.10, top_k=None):
    """
    Select the most important features in a graph.
    """

    node_mask = result["node_mask"]

    if node_mask is None:
        raise ValueError("Explanation has no node mask")

    importance = get_feature_importance(node_mask)

    F = len(importance)

    if top_k is not None:
        k = min(top_k, F)
    else:
        k = math.ceil(fraction * F)

    top_indices = torch.argsort(
        importance,
        descending=True
    )[:k]

    return {
        "row": result["row"],
        "label": result["label"],
        "pred": result["pred"],
        "selected_features": set(top_indices.tolist()),
        "feature_importance": importance
    }

def aggregate_important_features(results, selected_results, feature_names=None):
    graphs = []
    all_features = set()

    for original, selected in zip(results, selected_results):

        if original["row"] != selected["row"]:
            raise ValueError("Graph order does not match.")

        if original["label"] != original["pred"]:
            continue

        F = original["x"].shape[1]

        original_features = set(range(F))

        selected_features = selected["selected_features"]

        all_features.update(selected_features)

        graphs.append({
            "label": original["label"],
            "original_features": original_features,
            "selected_features": selected_features
        })

    rows = []

    for feature in sorted(all_features):

        row = {
            "feature": feature,
            "feature_name": (
                feature_names[feature]
                if feature_names is not None
                else f"feature_{feature}"
            )
        }

        for cls, name in [(0, "human"), (1, "machine")]:

            group = [
                g for g in graphs
                if g["label"] == cls
            ]

            present = sum(
                feature in g["original_features"]
                for g in group
            )

            selected = sum(
                feature in g["selected_features"]
                for g in group
            )

            row[f"{name}_present"] = present
            row[f"{name}_selected"] = selected

            row[f"{name}_rate"] = (
                selected / present
                if present > 0
                else float("nan")
            )

        row["difference"] = (
            row["machine_rate"] - row["human_rate"]
        )

        rows.append(row)

    return pd.DataFrame(rows)

def analyze_important_features(
    results,
    folder,
    path_base,
    feature_names=None,
    fraction=0.10,
    top_k=None
):
    os.makedirs(folder, exist_ok=True)

    fraction_name = (
        f"k{top_k}ft" if top_k is not None
        else f"{fraction:.2f}ft".replace(".", "")
    )

    selected_results = [
        select_top_features(
            result,
            fraction=fraction,
            top_k=top_k
        )
        for result in results
    ]

    feature_df = aggregate_important_features(
        results,
        selected_results,
        feature_names=feature_names
    )

    selected_path = os.path.join(
        folder,
        f"{path_base}_sel_feat{fraction_name}.pth"
    )

    feature_df_path = os.path.join(
        folder,
        f"{path_base}fc{fraction_name}.csv"
    )

    torch.save(selected_results, selected_path)

    feature_df.to_csv(
        feature_df_path,
        index=False
    )

    print(f"Unique important features: {len(feature_df)}")

    return feature_df