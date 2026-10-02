"""
Shared skeleton for every graph-level classifier in models/:

    node features --[conv -> norm -> ReLU -> dropout] x num_layers--> node embeddings
                  --readout (mean / sum / max / meanmax)--> graph embedding
                  --MLP head--> 2 logits (0 = human, 1 = machine)

A model only has to say how to build one conv layer and how to call it (see models/).

forward(x, edge_index, edge_weight=None, batch=None) is deliberately the signature that
torch_geometric.explain.Explainer / GNNExplainer expects, so trained checkpoints can be
explained directly
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool


class Readout(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        if kind not in ("mean", "sum", "max", "meanmax"):
            raise ValueError(f"unknown readout '{kind}'")
        self.kind = kind
        self.mult = 2 if kind == "meanmax" else 1

    def forward(self, h, batch):
        if self.kind == "mean":
            return global_mean_pool(h, batch)
        if self.kind == "sum":
            return global_add_pool(h, batch)
        if self.kind == "max":
            return global_max_pool(h, batch)
        return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)


class GraphClassifier(nn.Module):
    uses_edge_weight = True     # set False in subclasses whose conv ignores edge weights
    has_root_option = True      # set False where the conv already has its own root/self weight

    def __init__(self, args):
        super().__init__()
        self.args = args
        in_dim, hid = args.in_dim, args.hidden_dim
        self.use_weights = self.uses_edge_weight and args.edge_weight != "binary"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.roots = nn.ModuleList()
        # Separate "root" weight for the node's own features: h_i <- conv(h)_i + W_root h_i.
        # GCN/GAT/GIN otherwise push a node's own one-hot and its neighbours' one-hots through
        # the same matrix, so "the -> of" and "of -> the" become almost indistinguishable and
        # word-ORDER information is lost (SAGE and MLP already have this term).
        self.use_root = bool(args.root_weight) and self.has_root_option
        for layer in range(args.num_layers):
            d_in = in_dim if layer == 0 else hid
            self.convs.append(self.make_conv(d_in, hid, layer))
            self.norms.append(nn.BatchNorm1d(hid) if args.norm == "batch" else nn.Identity())
            self.roots.append(nn.Linear(d_in, hid, bias=False) if self.use_root else nn.Identity())
        self.dropout = nn.Dropout(args.dropout)
        self.readout = Readout(args.readout)
        g_dim = hid * self.readout.mult
        # Normalising the pooled graph vector keeps 'sum' readout from exploding (117 nodes per
        # graph) and keeps 'mean' readout from vanishing; without it training often collapses
        # to a constant prediction.
        self.graph_norm = nn.BatchNorm1d(g_dim) if args.norm == "batch" else nn.LayerNorm(g_dim)
        self.head = nn.Sequential(
            nn.Linear(g_dim, hid),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hid, args.num_classes),
        )

    # --- to implement in models/*.py -------------------------------------------------
    def make_conv(self, in_dim: int, out_dim: int, layer: int) -> nn.Module:
        raise NotImplementedError

    def conv_forward(self, conv, h, edge_index, edge_weight):
        raise NotImplementedError
    # ----------------------------------------------------------------------------------

    def node_embeddings(self, x, edge_index, edge_weight=None):
        if not self.use_weights:
            edge_weight = None
        h = x
        for conv, norm, root in zip(self.convs, self.norms, self.roots):
            out = self.conv_forward(conv, h, edge_index, edge_weight)
            if self.use_root:
                out = out + root(h)
            h = self.dropout(torch.relu(norm(out)))
        return h

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:   # single graph (e.g. when called by an explainer)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        h = self.node_embeddings(x, edge_index, edge_weight)
        return self.head(self.graph_norm(self.readout(h, batch)))
