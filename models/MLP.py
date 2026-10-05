"""
No-graph control: the same network but every node is processed on its own (edges are
ignored), then pooled. Answers "does the WAN structure add anything beyond the node
features (function-word counts)?". If a GNN does not beat this, the edges are not helping.
"""
import torch.nn as nn

from layers.graph_classifier import GraphClassifier


class Model(GraphClassifier):
    uses_edge_weight = False
    has_root_option = False

    def make_conv(self, in_dim, out_dim, layer):
        return nn.Linear(in_dim, out_dim)

    def conv_forward(self, conv, h, edge_index, edge_weight):
        return conv(h)
