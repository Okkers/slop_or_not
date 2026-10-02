from models import GAT, GCN, GIN, MLP, SAGE

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "GIN": GIN,
    "SAGE": SAGE,
    "MLP": MLP,   # no-graph control
}
