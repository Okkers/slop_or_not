"""
Graph Isomorphism Network (Xu et al., 2019) - the standard "most expressive" message-passing
baseline for graph classification. With weighted edges it uses GINE (Hu et al., 2020),
which adds a projection of the edge weight to every message.
"""
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv

from layers.graph_classifier import GraphClassifier


class Model(GraphClassifier):
    def _weighted(self):
        return self.uses_edge_weight and self.args.edge_weight != "binary"

    def make_conv(self, in_dim, out_dim, layer):
        mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                            nn.Linear(out_dim, out_dim))
        if self._weighted():
            return GINEConv(mlp, train_eps=True, edge_dim=1)
        return GINConv(mlp, train_eps=True)

    def conv_forward(self, conv, h, edge_index, edge_weight):
        if isinstance(conv, GINEConv):
            return conv(h, edge_index, edge_attr=edge_weight.view(-1, 1))
        return conv(h, edge_index)
