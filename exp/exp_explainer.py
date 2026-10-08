from torch_geometric.explain import Explainer, GNNExplainer

class GraphExplainer:
    def __init__(self, model, node_mask_type, edge_mask_type, epochs):
        model.eval()
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',
            node_mask_type=node_mask_type,
            edge_mask_type=edge_mask_type,
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',
                return_type='raw'
            ),
        )

    def explain(self, data):
        explanation = self.explainer(data.x, data.edge_index, edge_weight=data.edge_weight)    
        return explanation

    