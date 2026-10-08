import numpy as np

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
    total = edge_importance[idx].sum(axis=0) # summed importance per word pair
    count = edge_presence[idx].sum(axis=0) # number of essays with that word pair

    mean_all = total / n_excerpts # mean importance across all excerpts
    presence = count / n_excerpts # presence of that word pair across all excerpts

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_present = np.where(count > 0, total / count, np.nan) # mean importance across excerpts where that word pair is present

    return mean_all, mean_present, presence

def class_node_aggregation(word_importance, preds, cls):
    idx = preds == cls    
    if not idx.any():
        return np.full(word_importance.shape[1:], np.nan)
    return word_importance[idx].mean(axis=0)
    

    

