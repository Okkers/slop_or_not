import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import torch
import math
import pandas as pd
import os
import seaborn as sns
from vis.utils_node_feat_explainer import select_top_features


EDGE_GREY = "#b0b0b0"
NODE_GREY = "#c4c4c4"

def explanation_to_matrices(explanations, amount_nodes, amount_features):
    total_expl = len(explanations)
    edge_importance = np.zeros((total_expl, amount_nodes, amount_nodes))
    edge_presence = np.zeros((total_expl, amount_nodes, amount_nodes))

    node_feature_importance = np.zeros((total_expl, amount_nodes, amount_features))
    feature_importance = np.zeros((total_expl, amount_features))
    node_importance = np.zeros((total_expl, amount_nodes))

    for i, explanation in enumerate(explanations):
        src, dst = explanation["edge_index"].numpy()
        edge_presence[i, src, dst] = 1
        if explanation["edge_mask"] is not None:
            edge_importance[i, src, dst] = explanation["edge_mask"].numpy()
        if explanation["node_mask"] is not None:
            node_mask = explanation["node_mask"]
            if node_mask.shape == (amount_nodes, amount_features):
                node_feature_importance[i] = node_mask
                feature_importance[i] = node_mask.sum(axis=0)
                node_importance[i] = node_mask.sum(axis=1)

    return {
        "rows": np.array([explanation["row"] for explanation in explanations]),
        "labels": np.array([explanation["label"] for explanation in explanations]),
        "preds": np.array([explanation["pred"] for explanation in explanations]),
        "edge_importance": edge_importance,
        "edge_presence": edge_presence,
        "node_feature_importance": node_feature_importance,
        "feature_importance": feature_importance,
        "node_importance": node_importance
    }



def predict_with_edges(model, result, keep_edges, device):
    """
    Predict for trained model and the edges in keep_edges. (Useful for faithfulness analysis)
    """
    x = result["x"].to(device)

    edge_index = result["edge_index"][:, keep_edges].to(device)
    edge_weight = result["edge_weight"][keep_edges].to(device)

    batch = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=device
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            x,
            edge_index,
            edge_weight=edge_weight,
            batch=batch
        )

        probabilities = torch.softmax(logits, dim=-1)

    return probabilities[0].cpu()

def predict_with_edges_features(
    model,
    result,
    keep_edges,
    keep_features,
    device
):
    """
    Predict using selected edges and features.

    keep_edges: Boolean mask [E]
    keep_features: Boolean mask [F]

    Unselected edges are removed.
    Unselected feature values are set to zero for all nodes.
    """

    x = result["x"].to(device).clone()

    keep_edges = torch.as_tensor(
        keep_edges, dtype=torch.bool, device=device
    )

    keep_features = torch.as_tensor(
        keep_features, dtype=torch.bool, device=device
    )

    # Mask features
    x[:, ~keep_features] = 0.0

    # Select edges and their original WAN weights
    edge_index = result["edge_index"].to(device)[:, keep_edges]
    edge_weight = result["edge_weight"].to(device)[keep_edges]

    batch = torch.zeros(
        x.shape[0],
        dtype=torch.long,
        device=device
    )

    model.eval()

    with torch.no_grad():
        logits = model(
            x,
            edge_index,
            edge_weight=edge_weight,
            batch=batch
        )

        probabilities = torch.softmax(logits, dim=-1)

    return probabilities[0].cpu()

def evaluate_edge_faithfulness(results, folder, path_base, model, device, seed, fraction=0.05, top_k=None, random_repeats=100):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []

    for result in results:
        if result["label"] != result["pred"]:
            continue

        edge_index = result["edge_index"]
        edge_mask = result["edge_mask"]

        if edge_mask is None:
            raise ValueError("Explanation has no edge mask")

        E = edge_index.shape[1]
        eligible = torch.where(edge_index[0] != edge_index[1])[0]

        if top_k is not None:
            k = min(top_k, len(eligible))
        else:
            k = math.ceil(fraction * len(eligible))

        if k == 0:
            continue

        # Original predictions
        keep_all = torch.ones(E, dtype=torch.bool)
        prob_original = predict_with_edges(model, result, keep_all, device)
        predicted_class_original = int(prob_original.argmax())
        if predicted_class_original != result["pred"]:
            raise ValueError(f"Prediction mismatch for graph {result['row']}")
        original_confidence = prob_original[predicted_class_original].item()

        # Remove k GNNExplainer edges

        ## sufficiency
        sorted_indices = torch.argsort(edge_mask[eligible], descending=True)
        top_indices = eligible[sorted_indices[:k]]
        keep_only_top = torch.zeros(E, dtype=torch.bool)
        keep_only_top[top_indices] = True
        prob_only_top = predict_with_edges(model, result, keep_only_top, device)
        only_top_confidence = prob_only_top[predicted_class_original].item()
        sufficiency = original_confidence - only_top_confidence
        flipped_only_top = int(prob_only_top.argmax().item() != predicted_class_original)

        ## comprehensiveness
        keep_top = torch.ones(E, dtype=torch.bool)
        keep_top[top_indices] = False
        prob_top_removed = predict_with_edges(model, result, keep_top, device)
        top_removed_confidence = prob_top_removed[predicted_class_original].item()
        drop_original_top = original_confidence - top_removed_confidence
        flipped_original_top = int(prob_top_removed.argmax().item() != predicted_class_original)

        # Randomly remove k edges
        random_drops = []
        random_flips = []

        random_sufficiencies = []
        random_sufficiency_flips = []

        for _ in range(random_repeats):
            random_indices = rng.choice(
                eligible.numpy(),
                size=k,
                replace=False
            )

            ## comprehensiveness
            keep_random = torch.ones(E, dtype=torch.bool)
            keep_random[random_indices] = False
            prob_random = predict_with_edges(model, result, keep_random, device)
            random_confidence = prob_random[predicted_class_original].item()
            random_drops.append(original_confidence-random_confidence)
            random_flips.append(int(prob_random.argmax().item() != predicted_class_original))

            ## sufficiency
            keep_only_random = torch.zeros(E, dtype=torch.bool)
            keep_only_random[random_indices] = True
            prob_only_random = predict_with_edges(model, result, keep_only_random, device)
            random_only_confidence = prob_only_random[predicted_class_original].item()
            random_sufficiencies.append(original_confidence - random_only_confidence)
            random_sufficiency_flips.append(int(prob_only_random.argmax().item() != predicted_class_original))

        rows.append({
            "row": result["row"],
            "class": "Human" if result["label"] == 0 else "Machine",
            "original_confidence": original_confidence,

            # Comprehensiveness
            "comprehensiveness": drop_original_top,
            "comprehensiveness_random": np.mean(random_drops),

            # Sufficiency 
            "sufficiency": sufficiency,
            "sufficiency_random": np.mean(random_sufficiencies),

            # prediction flips
            "flip_top_removed": flipped_original_top,
            "flip_random_removed": np.mean(random_flips),

            "flip_only_top": flipped_only_top,
            "flip_only_random": np.mean(random_sufficiency_flips)
        })

    fraction_name = (
            f"k{top_k}" if top_k is not None
            else f"{fraction:.2f}".replace(".","")
        )
    df_faithfulness = pd.DataFrame(rows)
    faithfulness_path = os.path.join(
        folder,
        f"{path_base}edgeff{fraction_name}_{random_repeats}.csv"
    )
    df_faithfulness.to_csv(
        faithfulness_path,
        index=False
    )
    return df_faithfulness

def evaluate_edge_feature_faithfulness(
    results,
    folder,
    path_base,
    model,
    device,
    seed,
    edge_fraction=0.05,
    edge_top_k=None,
    feature_fraction=0.10,
    feature_top_k=None,
    random_repeats=100
):
    """
    Evaluate faithfulness of edge and feature explanations.

    Experiments:
    1. Edge-only
    2. Feature-only
    3. Edge + Feature
    4. Random edges + random features

    Sufficiency:
        Original confidence - confidence using only selected components.

    Comprehensiveness:
        Original confidence - confidence after removing selected components.

    Lower sufficiency is better.
    Higher comprehensiveness is better.
    """

    os.makedirs(folder, exist_ok=True)

    if random_repeats < 1:
        raise ValueError("random_repeats must be at least 1")

    rng = np.random.default_rng(seed)

    rows = []

    for result in results:

        if result["label"] != result["pred"]:
            continue

        edge_index = result["edge_index"].detach().cpu()
        edge_mask = result["edge_mask"]
        node_mask = result["node_mask"]

        if edge_mask is None or node_mask is None:
            raise ValueError("Explanation requires edge and node masks")

        edge_mask = torch.as_tensor(
            edge_mask
        ).detach().cpu().flatten()

        node_mask = torch.as_tensor(
            node_mask
        ).detach().cpu()

        N, F = result["x"].shape
        E = edge_index.shape[1]

        if node_mask.shape != (N, F):
            raise ValueError("Expected node-feature mask [N, F]")

        if edge_mask.numel() != E:
            raise ValueError("Edge mask size does not match edge_index")

        # ----------------------------------------
        # STEP 1: Select important edges
        # ----------------------------------------

        eligible = torch.where(
            edge_index[0] != edge_index[1]
        )[0]

        if edge_top_k is not None:
            k_edges = min(edge_top_k, len(eligible))
        else:
            k_edges = math.ceil(edge_fraction * len(eligible))

        if k_edges <= 0:
            continue

        sorted_indices = torch.argsort(
            edge_mask[eligible],
            descending=True
        )

        top_edge_indices = eligible[
            sorted_indices[:k_edges]
        ]

        # Boolean masks
        all_edges = torch.ones(E, dtype=torch.bool)

        top_edges = torch.zeros(E, dtype=torch.bool)
        top_edges[top_edge_indices] = True

        # ----------------------------------------
        # STEP 2: Select important features
        # ----------------------------------------

        # Uses mean feature importance across ALL nodes
        selected_features_result = select_top_features(
            result,
            fraction=feature_fraction,
            top_k=feature_top_k
        )

        top_feature_indices = sorted(
            selected_features_result["selected_features"]
        )

        k_features = len(top_feature_indices)

        if k_features == 0:
            continue

        all_features = torch.ones(F, dtype=torch.bool)

        top_features = torch.zeros(F, dtype=torch.bool)
        top_features[top_feature_indices] = True

        # ----------------------------------------
        # STEP 3: Original prediction
        # ----------------------------------------

        prob_original = predict_with_edges_features(
            model,
            result,
            all_edges,
            all_features,
            device
        )

        predicted_class_original = int(
            prob_original.argmax().item()
        )

        if predicted_class_original != result["pred"]:
            raise ValueError(
                f"Prediction mismatch for graph {result['row']}"
            )

        original_confidence = prob_original[
            predicted_class_original
        ].item()

        # ----------------------------------------
        # Helper for evaluating an intervention
        # ----------------------------------------

        def evaluate(keep_e, keep_f):

            probabilities = predict_with_edges_features(
                model,
                result,
                keep_e,
                keep_f,
                device
            )

            confidence = probabilities[
                predicted_class_original
            ].item()

            flipped = int(
                probabilities.argmax().item()
                != predicted_class_original
            )

            return confidence, flipped

        # ----------------------------------------
        # STEP 4: Edge-only
        # ----------------------------------------

        # Sufficiency: only important edges, all features
        edge_only_conf, edge_only_flip = evaluate(
            top_edges,
            all_features
        )

        # Comprehensiveness: remove important edges
        edge_removed_conf, edge_removed_flip = evaluate(
            ~top_edges,
            all_features
        )

        # ----------------------------------------
        # STEP 5: Feature-only
        # ----------------------------------------

        # Sufficiency: all edges, only important features
        feature_only_conf, feature_only_flip = evaluate(
            all_edges,
            top_features
        )

        # Comprehensiveness: remove important features
        feature_removed_conf, feature_removed_flip = evaluate(
            all_edges,
            ~top_features
        )

        # ----------------------------------------
        # STEP 6: Edge + Feature
        # ----------------------------------------

        # Sufficiency: only important edges AND features
        both_only_conf, both_only_flip = evaluate(
            top_edges,
            top_features
        )

        # Comprehensiveness: remove important edges AND features
        both_removed_conf, both_removed_flip = evaluate(
            ~top_edges,
            ~top_features
        )

        # ----------------------------------------
        # STEP 7: Random baseline
        # ----------------------------------------

        random_drops = []
        random_flips = []

        random_sufficiencies = []
        random_sufficiency_flips = []

        for _ in range(random_repeats):

            # Random edges (excluding self-loops)
            random_edge_indices = rng.choice(
                eligible.numpy(),
                size=k_edges,
                replace=False
            )

            # Random features
            random_feature_indices = rng.choice(
                F,
                size=k_features,
                replace=False
            )

            # Random masks
            random_edges = torch.zeros(
                E, dtype=torch.bool
            )
            random_edges[random_edge_indices] = True

            random_features = torch.zeros(
                F, dtype=torch.bool
            )
            random_features[random_feature_indices] = True

            # Comprehensiveness
            random_removed_conf, random_removed_flip = evaluate(
                ~random_edges,
                ~random_features
            )

            random_drops.append(
                original_confidence - random_removed_conf
            )
            random_flips.append(random_removed_flip)

            # Sufficiency
            random_only_conf, random_only_flip = evaluate(
                random_edges,
                random_features
            )

            random_sufficiencies.append(
                original_confidence - random_only_conf
            )

            random_sufficiency_flips.append(
                random_only_flip
            )

        # ----------------------------------------
        # STEP 8: Store results
        # ----------------------------------------

        rows.append({
            "row": result["row"],

            "class": (
                "Human" if result["label"] == 0
                else "Machine"
            ),

            "original_confidence": original_confidence,

            "num_edges_selected": k_edges,
            "num_features_selected": k_features,

            # Edge-only
            "sufficiency_edge": (
                original_confidence - edge_only_conf
            ),
            "comprehensiveness_edge": (
                original_confidence - edge_removed_conf
            ),

            # Feature-only
            "sufficiency_feature": (
                original_confidence - feature_only_conf
            ),
            "comprehensiveness_feature": (
                original_confidence - feature_removed_conf
            ),

            # Edge + Feature
            "sufficiency": (
                original_confidence - both_only_conf
            ),
            "comprehensiveness": (
                original_confidence - both_removed_conf
            ),

            # Random Edge + Feature
            "sufficiency_random": np.mean(
                random_sufficiencies
            ),
            "comprehensiveness_random": np.mean(
                random_drops
            ),

            # Prediction flips: Edge
            "flip_only_edges": edge_only_flip,
            "flip_edges_removed": edge_removed_flip,

            # Prediction flips: Features
            "flip_only_features": feature_only_flip,
            "flip_features_removed": feature_removed_flip,

            # Prediction flips: Both
            "flip_only_top": both_only_flip,
            "flip_top_removed": both_removed_flip,

            # Random flips
            "flip_only_random": np.mean(
                random_sufficiency_flips
            ),
            "flip_random_removed": np.mean(
                random_flips
            )
        })

    # ----------------------------------------
    # STEP 9: Save results
    # ----------------------------------------

    edge_name = (
        f"k{edge_top_k}"
        if edge_top_k is not None
        else f"{edge_fraction:.2f}".replace(".", "")
    )

    feature_name = (
        f"k{feature_top_k}"
        if feature_top_k is not None
        else f"{feature_fraction:.2f}".replace(".", "")
    )

    df_faithfulness = pd.DataFrame(rows)

    faithfulness_path = os.path.join(
        folder,
        f"{path_base}ff{edge_name}{feature_name}{random_repeats}.csv"
    )

    df_faithfulness.to_csv(
        faithfulness_path,
        index=False
    )

    print(f"Graphs evaluated: {len(df_faithfulness)}")
    print(f"Results saved to: {faithfulness_path}")

    return df_faithfulness

def plot_edge_faithfulness(df):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Comprehensiveness
    comp_df = df.melt(
        id_vars=["row", "class"],
        value_vars=[
            "comprehensiveness",
            "comprehensiveness_random"
        ],
        var_name="Method",
        value_name="Score"
    )

    comp_df["Method"] = comp_df["Method"].map({
        "comprehensiveness": "GNNExplainer",
        "comprehensiveness_random": "Random",
    })

    sns.boxplot(
        data=comp_df,
        x="class",
        y="Score",
        hue="Method",
        ax=axes[0]
    )

    axes[0].set_title("Comprehensiveness (Higher is Better)")
    axes[0].set_ylabel("Confidence drop")

    # Sufficiency
    suff_df = df.melt(
        id_vars=["row", "class"],
        value_vars=[
            "sufficiency",
            "sufficiency_random"
        ],
        var_name="Method",
        value_name="Score"
    )

    suff_df["Method"] = suff_df["Method"].map({
        "sufficiency": "GNNExplainer",
        "sufficiency_random": "Random",
    })

    sns.boxplot(
        data=suff_df,
        x="class",
        y="Score",
        hue="Method",
        ax=axes[1]
    )

    axes[1].set_title("Sufficiency (Lower is Better)")
    axes[1].set_ylabel("Confidence drop")

    plt.tight_layout()
    plt.show()

def _reduce(mask, n_items):
    if mask is None:
        return None
    m = mask.detach().abs().numpy() if hasattr(mask, "detach") else np.abs(np.asarray(mask))
    if m.shape[0] != n_items:
        return None
    return m.reshape(n_items, -1).sum(axis=1)


def _normalize(v):
    return v / v.max() if v is not None and v.max() > 0 else v


def build_explanation_graph(result):
    n = result["x"].shape[0]
    ei = result["edge_index"].numpy()
    ew = result["edge_weight"].numpy()

    node_imp = _normalize(_reduce(result["node_mask"], n))
    edge_imp = _normalize(_reduce(result["edge_mask"], ei.shape[1]))

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, importance=0.0 if node_imp is None else float(node_imp[i]))

    for k, (u, v) in enumerate(ei.T):
        u, v = int(u), int(v)
        if u == v:
            continue
        imp = 0.0 if edge_imp is None else float(edge_imp[k])
        if G.has_edge(u, v):
            G[u][v]["importance"] = max(G[u][v]["importance"], imp)
            G[u][v]["weight"] = max(G[u][v]["weight"], float(ew[k]))
        else:
            G.add_edge(u, v, importance=imp, weight=float(ew[k]))
    return G


def plot_explanation(result, threshold=0.5, pos=None, labels=None, dim=3, node_size=80, seed=0, title=None, show=True):
    
    G = build_explanation_graph(result)
    if pos is None:
        pos = nx.spring_layout(G, dim=dim, weight="weight", seed=seed)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)

    imp_edges = [(u, v) for u, v, d in G.edges(data=True) if d["importance"] >= threshold]
    other_edges = [(u, v) for u, v, d in G.edges(data=True) if d["importance"] < threshold]
    imp_nodes = [n for n, d in G.nodes(data=True) if d["importance"] >= threshold]
    other_nodes = [n for n in G.nodes if n not in set(imp_nodes)]

    def draw_edges(edges, color, lw, alpha, z):
        if not edges:
            return
        if dim == 3:
            segs = [(pos[u], pos[v]) for u, v in edges]
            ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha))
        else:
            for u, v in edges:
                ax.plot(*zip(pos[u], pos[v]), color=color, lw=lw, alpha=alpha, zorder=z)

    def draw_nodes(nodes, color, s, z):
        if not nodes:
            return
        xyz = np.array([pos[n] for n in nodes])
        ax.scatter(*xyz.T, c=color, s=s, depthshade=False, zorder=z)

    draw_edges(other_edges, "lightgrey", 0.6, 0.6, 1)
    draw_edges(imp_edges, "black", 2.0, 1.0, 2)
    draw_nodes(other_nodes, "lightgrey", node_size, 3)
    draw_nodes(imp_nodes, "black", node_size, 4)

    if labels is not None:
        for n, (p) in pos.items():
            ax.text(*p, labels.get(n, str(n)), fontsize=8,
                    color="black" if n in imp_nodes else "grey")

    ax.set_axis_off()
    ax.set_title(title or f"row {result['row']} | label {result['label']} | pred {result['pred']} "
                          f"(p_machine={result['prob_machine']:.2f}) | threshold {threshold}")
    if dim == 3:
        ax.auto_scale_xyz(*np.array(list(pos.values())).T)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax, pos

def _draw_edges(ax, pos, edges, color, lw, dim):
    if not edges:
        return
    rgba = to_rgba(color)
    if dim == 3:
        segs = [(pos[u], pos[v]) for u, v in edges]
        ax.add_collection3d(Line3DCollection(segs, colors=[rgba], linewidths=lw))
    else:
        lws = lw if isinstance(lw, (list, np.ndarray)) else [lw] * len(edges)
        for (u, v), w in zip(edges, lws):
            ax.plot(*zip(pos[u], pos[v]), color=rgba, lw=w)


def _draw_nodes(ax, pos, nodes, color, size):
    if not nodes:
        return
    xyz = np.array([pos[n] for n in nodes])
    ax.scatter(*xyz.T, color=to_rgba(color), s=size, depthshade=False)


def _finish_axes(ax, pos, dim):
    coords = np.array(list(pos.values()))
    if dim == 3:
        lo, hi = coords.min(0), coords.max(0)
        pad = 0.05 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def draw_original(ax, G, pos, dim=3, node_size=60):
    edges = list(G.edges)
    w = np.array([G[u][v]["weight"] for u, v in edges]) if edges else np.array([])
    lws = list(0.5 + 2.5 * w / w.max()) if len(w) and w.max() > 0 else 1.0
    _draw_edges(ax, pos, edges, "#555555", lws, dim)
    _draw_nodes(ax, pos, list(G.nodes), "#555555", node_size)
    _finish_axes(ax, pos, dim)
    ax.set_title(f"Input graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def draw_explanation(ax, G, pos, threshold, dim=3, node_size=60, labels=None):
    imp_edges = [(u, v) for u, v, d in G.edges(data=True) if d["importance"] >= threshold]
    oth_edges = [(u, v) for u, v, d in G.edges(data=True) if d["importance"] < threshold]
    imp_nodes = [n for n, d in G.nodes(data=True) if d["importance"] >= threshold]
    imp_set = set(imp_nodes)
    oth_nodes = [n for n in G.nodes if n not in imp_set]

    _draw_edges(ax, pos, oth_edges, EDGE_GREY, 1.0, dim)
    _draw_edges(ax, pos, imp_edges, "black", 2.5, dim)
    _draw_nodes(ax, pos, oth_nodes, NODE_GREY, node_size)
    _draw_nodes(ax, pos, imp_nodes, "black", node_size)

    if labels is not None:
        for n, p in pos.items():
            ax.text(*p, labels.get(n, str(n)), fontsize=8,
                    color="black" if n in imp_set else "grey")

    _finish_axes(ax, pos, dim)
    ax.set_title(f"Explanation (threshold {threshold}): "
                 f"{len(imp_nodes)} nodes, {len(imp_edges)} edges kept")


def _sync_views(fig, axes):
    def on_move(event):
        if event.inaxes in axes and event.button is not None:
            src = event.inaxes
            for a in axes:
                if a is not src:
                    a.view_init(elev=src.elev, azim=src.azim)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect("motion_notify_event", on_move)


def plot_comparison(result, threshold=0.5, pos=None, labels=None, dim=3,
                    node_size=60, seed=0, show=True):
    G = build_explanation_graph(result)
    if pos is None:
        pos = nx.spring_layout(G, dim=dim, weight="weight", seed=seed)

    kw = {"projection": "3d"} if dim == 3 else {}
    fig = plt.figure(figsize=(16, 7.5))
    ax1 = fig.add_subplot(121, **kw)
    ax2 = fig.add_subplot(122, **kw)

    draw_original(ax1, G, pos, dim, node_size)
    draw_explanation(ax2, G, pos, threshold, dim, node_size, labels)

    fig.suptitle(f"row {result['row']} | label {result['label']} | pred {result['pred']} "
                 f"(p_machine={result['prob_machine']:.2f})")
    if dim == 3:
        _sync_views(fig, [ax1, ax2])
    plt.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2), pos