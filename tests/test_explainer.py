"""
Checks that every GNN in models/ works with PyG's GNNExplainer for graph classification,
and shows how the aligned WAN node order turns per-excerpt edge masks into one
117 x 117 class-level "fingerprint" (RQ2 / E4).

Run:  uv run python -m tests.test_explainer        (needs data/sanity_wan.csv or set DATA=...)
"""
import os
from argparse import Namespace

import torch
from torch_geometric.explain import Explainer, GNNExplainer

from data_factory.data_loader import Dataset_MTT_Detection
from models import MODELS

DATA = os.environ.get("DATA", "data/sanity_wan.csv")


def make_args(model):
    return Namespace(
        model=model, data_path=DATA, function_words="data/function_words.txt", graph_type="wan",
        wan_source="text", window=5, node_features="count,freq,onehot", extra_node_cols="", edge_weight="wan",
        undirected=False, cache_dir="data/cache", use_cache=True, split="random", split_seed=2026,
        val_ratio=0.15, test_ratio=0.15, group_col="task_prompt", test_groups="", gen_col="generator",
        test_generators="", split_dir="splits", split_file="", num_layers=2, hidden_dim=32,
        dropout=0.0, readout="sum", norm="batch", root_weight=True, heads=4, attn_dropout=0.0,
        gat_v2=False, num_classes=2)


def explain_graphs(model, graphs, epochs=30):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="phenomenon",         # explain the TRUE label
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    n = graphs[0].num_nodes
    fingerprint = torch.zeros(n, n)
    node_importance = torch.zeros(n)
    for g in graphs:
        exp = explainer(g.x, g.edge_index, target=g.y, edge_weight=g.edge_weight)
        assert exp.edge_mask.shape[0] == g.edge_index.shape[1]
        # node i is the same function word in every graph, so masks can simply be summed
        fingerprint.index_put_((g.edge_index[0], g.edge_index[1]), exp.edge_mask, accumulate=True)
        node_importance += exp.node_mask.sum(dim=1)
    return fingerprint / len(graphs), node_importance / len(graphs)


def test_explainer_all_models():
    torch.manual_seed(0)
    data = Dataset_MTT_Detection(make_args("GCN"), "test")
    graphs = data.graphs[:3]
    for name in ("GCN", "GAT", "GIN", "SAGE"):
        args = make_args(name)
        args.in_dim = graphs[0].num_node_features
        model = MODELS[name].Model(args).eval()
        fp, imp = explain_graphs(model, graphs, epochs=10)
        assert fp.shape == (117, 117) and imp.shape == (117,)
        assert torch.isfinite(fp).all()
        print(f"{name}: GNNExplainer OK, top fingerprint edge weight {fp.max():.3f}")
    args = make_args("GAT")
    args.in_dim = graphs[0].num_node_features
    gat = MODELS["GAT"].Model(args).eval()
    att = gat.attention_weights(graphs[0].x, graphs[0].edge_index, graphs[0].edge_weight)
    assert len(att) == 2 and att[0][1].shape[1] == 4
    print("GAT attention weights OK:", [tuple(a.shape) for _, a in att])


if __name__ == "__main__":
    test_explainer_all_models()
