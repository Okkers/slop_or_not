from models import GAT, GCN, MLP, SAGE

MODELS = {
    "GCN": GCN,
    "GAT": GAT,
    "SAGE": SAGE,
    "MLP": MLP,   # no-graph control
}
