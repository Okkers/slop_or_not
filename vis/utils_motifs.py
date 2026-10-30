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
from vis.utils_explainer import select_top_edges

def detect_two_edge_motifs(edge_index, max_motifs=200_000):
    """
    Detects motifs of size 3 (3 nodes, 2 edges)
    """
    edges = {
        (int(u), int(v))
        for u, v in edge_index.detach().cpu().T.tolist()
        # if u != v
    }

    incident = defaultdict(set)
    for u, v in edges:
        incident[u].add((u,v))
        incident[v].add((u,v))

    motifs = set()

    for node_edges in incident.values():
        for e1, e2 in combinations(node_edges, 2):
            motifs.add(tuple(sorted((e1,e2))))

            if len(motifs) > max_motifs:
                raise ValueError("Too many 2-edge motifs")
    return motifs

def aggregate_motifs(results, selected_results):
    """
    Aggregates for every motif (of every machine and human graph).
    """
    graphs = []
    all_motifs = set()

    # find the original edges and motifs in the WAN
    for original, selected in zip(results, selected_results):
        if original["label"] != original["pred"]:
            continue

        original_edges = {
            (int(u), int(v))
            for u, v in original["edge_index"].T.tolist()
            # if u != v
        }

        motifs = detect_two_edge_motifs(selected["edge_index"])

        all_motifs.update(motifs)

        graphs.append({
            "label": original["label"],
            "original_edges": original_edges,
            "motifs": motifs
        })

    # compare every candidate motif between classes
    rows = []

    for motif in sorted(all_motifs):
        row = {"motif": motif}

        for cls, name in [(0, "human"), (1, "machine")]:
            group = [g for g in graphs if g["label"] == cls]

            # number of graphs where motif exists
            present = sum(
                all(edge in g["original_edges"] for edge in motif)
                for g in group
            )

            # number of graphs where motif is selected
            selected = sum(
                motif in g["motifs"]
                for g in group
            )

            row[f"{name}_present"] = present
            row[f"{name}_selected"] = selected

            row[f"{name}_rate"] = (
                selected / present if present > 0 else float("nan")
            )

        row["difference"] = (row["machine_rate"] - row["human_rate"])

        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            "difference",
            key=lambda x: x.abs(),
            ascending=False
        )

    return df

def test_motif_differences(motif_df, results, selected_results,
                           min_present=10):
    """
    Function answers 3 questions:
    1. Is the motif frequently selected in machine and human graphs?
    2. Is the difference between machine and human statistically significant? -> Fisher's exact test ("Is the motif selected at different rates 
    in human and machine graphs?")
    3. How often would the motif be selected randomly? -> if you select e.g. 5% of the edges, how likely is it that both edges
    belonging to the same motif are selected? ("Is the motif selected more often than random edge selection would suggest?")

    results: original results
    selected_results: results but with the most important selected edges instead of original edges
    """
    if len(results) != len(selected_results):
        raise ValueError("Results and selected_results must have equal length.")

    if motif_df.empty:
        return motif_df.copy()

    graphs = []

    for original, selected in zip(results, selected_results):

        # Collect information about each correctly classified graph
        if original["row"] != selected["row"]:
            raise ValueError("Graph order does not match.")

        if original["label"] != original["pred"]:
            continue

        # Extract the original and selected edges
        original_edges = {
            (int(u), int(v))
            for u, v in original["edge_index"].T.tolist()
            # if u != v
        }

        selected_edges = {
            (int(u), int(v))
            for u, v in selected["edge_index"].T.tolist()
            # if u != v
        }

        if not selected_edges.issubset(original_edges):
            raise ValueError("Selected edges must belong to the original WAN.")

        E = len(original_edges)
        k = len(selected_edges)

        # Probability that two particular edges are randomly selected
        # Is this motif selected more frequently than expected from random top-5% edge selection?
        # k(k-1) / E(E-1)
        p_random = (
            k * (k - 1) / (E * (E - 1))
            if E >= 2 else 0.0
        )

        graphs.append({
            "label": original["label"],
            "edges": original_edges,
            "p_random": p_random
        })

    df = motif_df.copy()

    df["human_random_rate"] = np.nan
    df["machine_random_rate"] = np.nan
    df["p_value"] = np.nan

    # Analyze every candidate motif
    for idx, row in df.iterrows():

        motif = set(row["motif"])

        for cls, name in [(0, "human"), (1, "machine")]:

            # Only graphs where the motif exists
            eligible = [
                g for g in graphs
                if g["label"] == cls
                and motif.issubset(g["edges"])
            ]

            # mean of prob random selection for all graphs containing the motif in class cls
            if eligible:
                df.at[idx, f"{name}_random_rate"] = np.mean(
                    [g["p_random"] for g in eligible]
                )

        # Observed selection counts
        h_present = int(row["human_present"])
        h_selected = int(row["human_selected"])

        m_present = int(row["machine_present"])
        m_selected = int(row["machine_selected"])

        # Require sufficient support in both classes
        if h_present < min_present or m_present < min_present:
            continue

        # Fisher's exact test: H_0: P_H(motif) = P_M(motif) -> when motif exists, it has same
        # probability of being selected in human and machine explanations
        table = [
            [h_selected, h_present - h_selected], # amount selected, amount not selected
            [m_selected, m_present - m_selected]  # amount selected, amount not selected
        ]

        _, p_value = fisher_exact(table)

        df.at[idx, "p_value"] = p_value

    # Benjamini-Hochberg correction
    valid = df["p_value"].notna()
    p = df.loc[valid, "p_value"].to_numpy()

    df["q_value"] = np.nan

    if len(p) > 0:
        order = np.argsort(p)

        q_sorted = (
            p[order] * len(p) / np.arange(1, len(p) + 1)
        )

        q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]

        q = np.empty_like(p)
        q[order] = np.clip(q_sorted, 0, 1)

        df.loc[valid, "q_value"] = q

    # Sort by significance, then magnitude of class difference
    df["abs_difference"] = df["difference"].abs()

    return df.sort_values(
        ["q_value", "abs_difference"],
        ascending=[True, False],
        na_position="last"
    )

def motif_to_fwords(motif, function_words):

    return (
        str(function_words[motif[0]]),
        str(function_words[motif[1]])
    )

def analyze_motifs(results, folder, path_base, fraction=0.05, top_k = None, min_present=10):
    """
    Saves:
    * Most important edges used to find the motifs.
    * All 2-edge motifs for all graphs seperately.
    * Aggregates of motifs found in human and machine graph.
    * Aggregates including statistical significance information.
    """

    os.makedirs(folder, exist_ok=True)

    fraction_name = (
        f"k{top_k}" if top_k is not None
        else f"{fraction:.2f}".replace(".","")
    )

    # Select the top fraction of important edges
    selected_results = [
        select_top_edges(result, fraction=fraction, top_k=top_k)
        for result in results
    ]

    # Detect 2-edge motfs
    motif_results = []
    for selected in selected_results:
        motifs = detect_two_edge_motifs(selected["edge_index"])
        motif_results.append({
            "row": selected["row"],
            "label": selected["label"],
            "pred": selected["pred"],
            "motifs": sorted(motifs),
            "num_motifs": len(motifs)
        })

    # Aggregate motifs across graphs
    motif_df = aggregate_motifs(results, selected_results)

    # Statistical testing
    motif_stats = test_motif_differences(
        motif_df,
        results,
        selected_results,
        min_present=min_present
    )

    # Save results
    edges_path = os.path.join(folder, f"{path_base}_motedg{fraction_name}.pth")
    motifs_path = os.path.join(folder, f"{path_base}_mot2edg{fraction_name}.pth")
    motif_df_path = os.path.join(folder, f"{path_base}_motcom{fraction_name}.csv")
    motif_stats_path = os.path.join(folder, f"{path_base}_motstat{fraction_name}.csv")

    torch.save(selected_results, edges_path)
    torch.save(motif_results, motifs_path)
    motif_df.to_csv(motif_df_path, index=False)
    motif_stats.to_csv(motif_stats_path, index=False)

    print(f"\nGraphs analyzed: {len(results)}")
    print(f"Unique motifs detected: {len(motif_df)}")

    return motif_stats

