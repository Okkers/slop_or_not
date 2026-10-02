"""GraphSAGE (Hamilton et al., 2017) with mean aggregation. Ignores edge weights (only uses which edges exist)."""
from torch_geometric.nn import SAGEConv

from layers.graph_classifier import GraphClassifier


class Model(GraphClassifier):
    uses_edge_weight = False
    has_root_option = False

    def make_conv(self, in_dim, out_dim, layer):
        return SAGEConv(in_dim, out_dim, aggr="mean")

    def conv_forward(self, conv, h, edge_index, edge_weight):
        return conv(h, edge_index)
