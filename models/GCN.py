"""Graph Convolutional Network (Kipf & Welling, 2017). Uses the WAN edge weights."""
from torch_geometric.nn import GCNConv

from layers.graph_classifier import GraphClassifier


class Model(GraphClassifier):
    def make_conv(self, in_dim, out_dim, layer):
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=True)

    def conv_forward(self, conv, h, edge_index, edge_weight):
        return conv(h, edge_index, edge_weight=edge_weight)
