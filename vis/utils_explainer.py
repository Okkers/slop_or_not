import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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

def class_edge_aggregation(edge_importance, edge_presence, preds, cls):
    idx = preds == cls
    n_excerpts = max(1, idx.sum()) 
    total = edge_importance[idx].sum(axis=0) 
    count = edge_presence[idx].sum(axis=0)

    mean_all = total / n_excerpts
    presence = count / n_excerpts

    mean_present = np.where(count > 0, total / count, np.nan)

    return mean_all, mean_present, presence

def class_node_aggregation(word_importance, preds, cls):
    idx = preds == cls    
    if not idx.any():
        return np.full(word_importance.shape[1:], np.nan)
    return word_importance[idx].mean(axis=0)


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

