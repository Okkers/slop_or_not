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
from collections import defaultdict
# from utils_motifs import detect_two_edge_motifs

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
        src, dst = explanation["edge_index"].cpu().numpy()
        edge_presence[i, src, dst] = 1

        if explanation["edge_mask"] is not None:
            edge_importance[i, src, dst] = explanation["edge_mask"].cpu().numpy()

        if explanation["node_mask"] is not None:
            node_mask = explanation["node_mask"].detach().cpu().numpy()

            if node_mask.shape == (amount_nodes, amount_features):
                # attributes: importance for node-feature pair
                node_feature_importance[i] = node_mask
                feature_importance[i] = node_mask.sum(axis=0)
                node_importance[i] = node_mask.sum(axis=1)

            elif node_mask.shape == (amount_nodes, 1):
                # objectL one important score per node
                node_importance[i] = node_mask[:, 0]

            elif node_mask.shape == (1, amount_features):
                # common_attributes: one importance score per feature
                feature_importance[i] = node_mask[0]

            else:
                raise ValueError(
                    f"Unexpected node mask shape: {node_mask.shape}"
                )

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

def get_node_importance(node_mask, node_mask_type="object"):
    """
    Convert node_mask into node importance if needed. 
    """
    if node_mask is None:
        return None

    mask = torch.as_tensor(node_mask).detach().cpu()

    if mask.ndim == 1:
        return mask

    if node_mask_type == "object":
        return mask[:,0]

    elif node_mask_type == "attributes":
        return mask.mean(dim=1)

    else:
        raise ValueError(
            f"Unsupported node_mask_type: {node_mask_type}"
        )

def select_top_nodes(result, fraction=0.10, top_k=None, node_mask_type="object"):
    """
    Same as select_top_edges, but now for nodes.
    """ 
    node_mask = result["node_mask"]
    if node_mask is None:
        raise ValueError("Explanation has no node mask")

    node_mask = torch.as_tensor(node_mask)

    importance = get_node_importance(node_mask, node_mask_type=node_mask_type)

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

def select_top_edges(result, fraction=0.05, top_k=None):
    """
    Selects the edges the GNNExplainer gives the highest weights to. 
    Fraction: selects fraction % of the edges to be considered important.
    top_k: selects the top k highest importance mask scores.
    """
    edge_index = result["edge_index"]
    edge_mask = result["edge_mask"]

    if edge_mask is None:
        raise ValueError("Explanation has no edge mask")

    # valid = edge_index[0] != edge_index[1]
    # edge_index = edge_index[:, valid]
    # edge_mask = edge_mask[valid]

    total_edges = edge_index.shape[1]

    if top_k is not None:
        k = min(top_k, total_edges)
    else:
        k = math.ceil(fraction * total_edges)

    sorted_indices = torch.argsort(edge_mask, descending=True)
    selected_indices = sorted_indices[:k]
    selected_edges = edge_index[:, selected_indices]
    selected_masks = edge_mask[selected_indices]

    return {
        "row": result["row"],
        "label": result["label"],
        "pred": result["pred"],
        "edge_index": selected_edges,
        "edge_masks_selected": selected_masks,
        "num_edges_original": total_edges,
        "num_edges_selected": k
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

def evaluate_edge_faithfulness(results, folder, path_base, model, device, seed, fraction=0.05, top_k=None, random_repeats=100):
    # os.makedirs(folder, exist_ok=True)
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
        # eligible = torch.where(edge_index[0] != edge_index[1])[0]
        eligible = torch.arange(E, device=edge_index.device)

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
            # random_indices = rng.choice(
            #     eligible.numpy(),
            #     size=k,
            #     replace=False
            # )
            random_indices = rng.choice(
                    E,
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

    # fraction_name = (
    #         f"k{top_k}" if top_k is not None
    #         else f"{fraction:.2f}".replace(".","")
    #     )
    df_faithfulness = pd.DataFrame(rows)
    # faithfulness_path = os.path.join(
    #     folder,
    #     f"{path_base}edgeff{fraction_name}_{random_repeats}.csv"
    # )
    # df_faithfulness.to_csv(
    #     faithfulness_path,
    #     index=False
    # )
    return df_faithfulness

def evaluate_edge_faithfulness_per_motif(results, folder, path_base, model, device, seed,
                               fraction=0.4, top_k=None, random_repeats=20):
    """
    One row per (graph, 2-edge motif). Each motif is perturbed on its own:
      - comprehensiveness: confidence drop when ONLY the motif's two edges are removed
      - sufficiency: confidence drop when ONLY the motif's two edges are kept
    Motifs come from detect_two_edge_motifs on the explainer's top edges of that graph.
    Controls: two random edges, and two random edges sharing a node.
    """
    # local import: utils_motifs imports select_top_edges from this module (circular otherwise)
    from vis.utils_motifs import detect_two_edge_motifs

    print(f"Evaluating edge faithfulness for {path_base}")

    rng = np.random.default_rng(seed)
    rows = []
    analyzed=0
    for result in results:
        print(f"Test sample {analyzed}")
        if result["label"] != result["pred"]:
            continue

        edge_index = result["edge_index"]
        edge_mask = result["edge_mask"]
        if edge_mask is None:
            raise ValueError("Explanation has no edge mask")

        E = edge_index.shape[1]
        if E < 2:
            continue

        # top edges and the 2-edge motifs inside them
        selected = select_top_edges(result, fraction=fraction, top_k=top_k)
        if selected["num_edges_selected"] < 2:
            continue
        motifs = sorted(detect_two_edge_motifs(selected["edge_index"]))
        if not motifs:
            continue

        # directed edge (u, v) -> indices in the ORIGINAL edge_index (list guards against duplicates)
        pairs = [tuple(p) for p in edge_index.T.tolist()]
        edge_idx = defaultdict(list)
        incident = defaultdict(set)
        for i, (u, v) in enumerate(pairs):
            edge_idx[(u, v)].append(i)
            incident[u].add(i)
            incident[v].add(i)

        # original prediction
        prob_original = predict_with_edges(model, result, torch.ones(E, dtype=torch.bool), device)
        c = int(prob_original.argmax())
        if c != result["pred"]:
            raise ValueError(f"Prediction mismatch for graph {result['row']}")
        base = prob_original[c].item()

        def evaluate(idx, keep_only):
            """(confidence drop, flipped) when removing idx, or keeping only idx."""
            keep = torch.zeros(E, dtype=torch.bool) if keep_only else torch.ones(E, dtype=torch.bool)
            keep[list(idx)] = keep_only
            prob = predict_with_edges(model, result, keep, device)
            return base - prob[c].item(), int(prob.argmax().item() != c)

        def random_pair():
            return rng.choice(E, size=2, replace=False).tolist()

        def random_adjacent_pair():
            for _ in range(50):
                i = int(rng.integers(E))
                u, v = pairs[i]
                cand = [j for j in (incident[u] | incident[v]) if j != i]
                if cand:
                    return [i, int(rng.choice(cand))]
            return random_pair()

        for motif in motifs:
            # print(f"Motif {motif}")
            idx = [i for e in motif for i in edge_idx[e]]

            # the motif alone, one at a time
            comp, flip_comp = evaluate(idx, keep_only=False)
            suff, flip_suff = evaluate(idx, keep_only=True)

            # controls
            rp_comp, rp_suff, ra_comp, ra_suff = [], [], [], []
            rp_fc, rp_fs, ra_fc, ra_fs = [], [], [], []
            for _ in range(random_repeats):
                a = random_pair()
                d, f = evaluate(a, False); rp_comp.append(d); rp_fc.append(f)
                d, f = evaluate(a, True);  rp_suff.append(d); rp_fs.append(f)

                b = random_adjacent_pair()
                d, f = evaluate(b, False); ra_comp.append(d); ra_fc.append(f)
                d, f = evaluate(b, True);  ra_suff.append(d); ra_fs.append(f)

            rows.append({
                "row": result["row"],
                "class": "Human" if result["label"] == 0 else "Machine",
                "motif": motif,
                "has_self_loop": int(any(u == v for u, v in motif)),
                "mean_edge_mask": float(edge_mask[idx].mean()),
                "original_confidence": base,

                # motif only
                "comprehensiveness": comp,
                "sufficiency": suff,
                "flip_removed": flip_comp,
                "flip_only": flip_suff,

                # control: two random edges
                "comprehensiveness_random_pair": np.mean(rp_comp),
                "sufficiency_random_pair": np.mean(rp_suff),
                "flip_removed_random_pair": np.mean(rp_fc),
                "flip_only_random_pair": np.mean(rp_fs),

                # control: two random edges sharing a node (the fair control for a motif)
                "comprehensiveness_random_adjacent": np.mean(ra_comp),
                "sufficiency_random_adjacent": np.mean(ra_suff),
                "flip_removed_random_adjacent": np.mean(ra_fc),
                "flip_only_random_adjacent": np.mean(ra_fs),
            })
        analyzed=analyzed+1

    df = pd.DataFrame(rows)

    os.makedirs(folder, exist_ok=True)
    fraction_name = (
        f"k{top_k}" if top_k is not None
        else f"{fraction:.2f}".replace(".", "")
    )
    path = os.path.join(
        folder,
        f"{path_base}_motfaith{fraction_name}_{random_repeats}.csv"
    )
    df.to_csv(path, index=False)
    print(f"Saved motif faithfulness to {path}")

    return df


def plot_edge_faithfulness(df, folder):

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

    figures_folder = os.path.join(folder, "figures")
    os.makedirs(figures_folder, exist_ok = True)

    fig.savefig(os.path.join(figures_folder, "edge_faith.png"), dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    
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