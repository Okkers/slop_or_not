"""
Graph Attention Network (Velickovic et al., 2018); --gat_v2 switches to GATv2 (Brody et al., 2022).
The WAN edge weight is fed to the attention as a 1-dim edge feature (edge_dim=1).

`attention_weights(...)` returns the per-layer attention coefficients, which E5 compares
against GNNExplainer masks.
"""
import torch
from torch_geometric.nn import GATConv, GATv2Conv

from layers.graph_classifier import GraphClassifier


class Model(GraphClassifier):
    def make_conv(self, in_dim, out_dim, layer):
        heads = self.args.heads
        if out_dim % heads:
            raise ValueError(f"--hidden_dim ({out_dim}) must be divisible by --heads ({heads})")
        Conv = GATv2Conv if self.args.gat_v2 else GATConv
        return Conv(in_dim, out_dim // heads, heads=heads, concat=True,
                    dropout=self.args.attn_dropout,
                    edge_dim=1 if self.uses_edge_weight and self.args.edge_weight != "binary" else None,
                    add_self_loops=True)

    def conv_forward(self, conv, h, edge_index, edge_weight, return_attention=False):
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        return conv(h, edge_index, edge_attr=edge_attr, return_attention_weights=return_attention or None)

    @torch.no_grad()
    def attention_weights(self, x, edge_index, edge_weight=None):
        """List over layers of (edge_index_with_self_loops, alpha[num_edges, heads])."""
        if not self.use_weights:
            edge_weight = None
        out, h = [], x
        for conv, norm, root in zip(self.convs, self.norms, self.roots):
            z, (ei, alpha) = self.conv_forward(conv, h, edge_index, edge_weight, return_attention=True)
            out.append((ei, alpha))
            if self.use_root:
                z = z + root(h)
            h = self.dropout(torch.relu(norm(z)))
        return out
